import logging
import json
from decimal import Decimal
import requests
import pyupbit
from django.conf import settings
from openai import OpenAI
from typing import Dict, Optional
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import base64
from datetime import datetime
import os
from .models import TradingRecord

logger = logging.getLogger(__name__)


class BitcoinAnalyzer:
    def __init__(self):
        # self.upbit = pyupbit.Upbit(
        #     settings.UPBIT_ACCESS_KEY,
        #     settings.UPBIT_SECRET_KEY
        # )
        self.upbit = None
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def capture_chart(self) -> Optional[str]:
        """캡처 차트 이미지를 base64로 인코딩하여 반환"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920x1080')

        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)
            wait = WebDriverWait(driver, 20)

            # 업비트 차트 페이지 로드
            driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-BTC")

            # 차트 로딩 대기
            chart_element = wait.until(
                EC.presence_of_element_located((By.ID, "fullChartiq"))
            )
            time.sleep(3)

            # 1초 단위 차트로 변경
            time_menu = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "cq-menu.ciq-period"))
            )
            time_menu.click()
            time.sleep(1)

            one_second_option = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "cq-item[stxtap*='second']"))
            )
            one_second_option.click()
            time.sleep(2)

            # MACD 추가
            study_menu = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "cq-menu.ciq-studies"))
            )
            study_menu.click()
            time.sleep(1)

            macd_item = wait.until(
                EC.presence_of_element_located((By.XPATH, "//cq-item[contains(., 'MACD')]"))
            )
            ActionChains(driver).move_to_element(macd_item).click().perform()
            time.sleep(1)

            # 볼린저 밴드 추가
            study_menu.click()
            time.sleep(1)

            bb_item = wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, "//cq-item[.//translate[@original='Bollinger Bands']]")
                )
            )
            ActionChains(driver).move_to_element(bb_item).click().perform()
            time.sleep(2)

            # 스크린샷 캡처 및 base64 인코딩
            screenshot = driver.get_screenshot_as_png()
            base64_image = base64.b64encode(screenshot).decode('utf-8')

            return base64_image

        except Exception as e:
            logger.error(f"Chart capture error: {e}")
            return None
        finally:
            if driver:
                driver.quit()

    def get_bitcoin_data(self, symbol='KRW-BTC', max_retries=3) -> Optional[Dict]:
        """Fetch Bitcoin data with technical indicators from API"""
        base_url = "https://gridtrade.one/api/v1/binanceData/upbit"
        session = requests.Session()
        for attempt in range(max_retries):
            try:
                response = session.get(
                    f"{base_url}/{symbol}/minute1/",
                    timeout=30,
                    verify=False,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                response.raise_for_status()
                return response.json()

            except Exception as e:
                logger.error(f"데이터 가져오기 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue
            finally:
                session.close()

        return None

    def generate_trade_reflection(self, previous_decisions: str, current_price: Decimal) -> str:
        """Generate reflection on previous trading decisions"""
        try:
            reflection_prompt = f"""
            As a trading advisor, analyze the previous trading decisions and current market price to provide a reflection.


            Evaluate recent scalping trades based on the following:
            
            1. **Effectiveness**: Were recent trades profitable and aligned with market trends?
            2. **Missed Opportunities**: Identify any missed signals or overtrading instances.
            3. **Improvements**: Suggest one quick adjustment for immediate strategy enhancement.
            
            Previous Decisions:
            {previous_decisions}

            Current BTC Price: {current_price}

            Format your response as a concise paragraph focusing on the most recent trades.
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a cryptocurrency trading advisor providing reflections on past trades."},
                    {"role": "user", "content": reflection_prompt}
                ],
                max_tokens=500  # 짧게 유지
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            return "반성 생성 중 오류 발생"
    def get_current_status(self) -> Dict:
        """Get current trading account status"""
        try:
            orderbook = pyupbit.get_orderbook(ticker="KRW-BTC")
            current_time = orderbook['timestamp']

            # 실제 Upbit 연동이 없을 경우의 기본값
            if self.upbit is None:
                return json.dumps({
                    'current_time': current_time,
                    'orderbook': orderbook,
                    'btc_balance': "0.0",
                    'krw_balance': "1000000.0",  # 100만원 기본값
                    'btc_avg_buy_price': "0.0"
                })

            # Upbit 연동이 있는 경우 실제 데이터 조회
            balances = self.upbit.get_balances()
            btc_balance = Decimal('0')
            krw_balance = Decimal('0')
            btc_avg_buy_price = Decimal('0')

            for b in balances:
                if b['currency'] == "BTC":
                    btc_balance = Decimal(b['balance'])
                    btc_avg_buy_price = Decimal(b['avg_buy_price'])
                if b['currency'] == "KRW":
                    krw_balance = Decimal(b['balance'])

            return json.dumps({
                'current_time': current_time,
                'orderbook': orderbook,
                'btc_balance': str(btc_balance),
                'krw_balance': str(krw_balance),
                'btc_avg_buy_price': str(btc_avg_buy_price)
            })
        except Exception as e:
            logger.error(f"Error getting current status: {e}")
            return None


    # def get_current_status(self) -> Dict:
    #     """Get current trading account status"""
    #     try:
    #         orderbook = pyupbit.get_orderbook(ticker="KRW-BTC")
    #         current_time = orderbook['timestamp']
    #         balances = self.upbit.get_balances()
    #
    #         btc_balance = Decimal('0')
    #         krw_balance = Decimal('0')
    #         btc_avg_buy_price = Decimal('0')
    #
    #         for b in balances:
    #             if b['currency'] == "BTC":
    #                 btc_balance = Decimal(b['balance'])
    #                 btc_avg_buy_price = Decimal(b['avg_buy_price'])
    #             if b['currency'] == "KRW":
    #                 krw_balance = Decimal(b['balance'])
    #
    #         return json.dumps({
    #             'current_time': current_time,
    #             'orderbook': orderbook,
    #             'btc_balance': str(btc_balance),
    #             'krw_balance': str(krw_balance),
    #             'btc_avg_buy_price': str(btc_avg_buy_price)
    #         })
    #     except Exception as e:
    #         logger.error(f"Error getting current status: {e}")
    #         return None

    def get_last_decisions(self, num_decisions: int = 10, current_price = 100000000) -> str:
        """Fetch recent trading decisions from database"""
        try:
            decisions = TradingRecord.objects.order_by('-created_at')[:num_decisions]

            formatted_decisions = []
            for decision in decisions:
                formatted_decision = {
                    "timestamp": int(decision.created_at.timestamp() * 1000),
                    "decision": decision.trade_type.lower(),
                    "percentage": float(decision.trade_ratio),
                    "reason": decision.trade_reason,
                    "btc_balance": float(decision.coin_balance),
                    "krw_balance": float(decision.balance),
                    "btc_avg_buy_price": float(decision.current_price),
                    "current_price": {current_price}

                }
                formatted_decisions.append(str(formatted_decision))

            return "\n".join(formatted_decisions)
        except Exception as e:
            logger.error(f"Error getting last decisions: {e}")
            return ""

    def analyze_with_gpt4(self, market_data: Dict, last_decisions: str, fear_and_greed, current_status: str) -> Optional[Dict]:
        """Analyze market data using GPT-4"""
        try:
            # 현재 파일의 디렉토리 경로를 가져옴
            import os

            current_dir = os.path.dirname(os.path.abspath(__file__))
            instructions_path = os.path.join(current_dir, 'instructions_v3.md')

            with open(instructions_path, 'r', encoding='utf-8') as file:
                instructions = file.read()

            # 차트 이미지 캡처
            chart_image = self.capture_chart()

            if not chart_image:
                logger.warning("Failed to capture chart image")

            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(market_data)},
                {"role": "user", "content": last_decisions},
                {"role": "user", "content": fear_and_greed},
                {"role": "user", "content": current_status}
            ]

            # 이미지가 성공적으로 캡처된 경우에만 추가
            if chart_image:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{chart_image}"
                            }
                        }
                    ]
                })

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # 이미지를 처리할 수 있는 모델로 변경
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=1000
            )


            result = json.loads(response.choices[0].message.content)
            print(str(result))
            print("----------------------------")
            # 응답 검증 및 기본값 설정
            if not isinstance(result, dict):
                raise ValueError("GPT response is not a dictionary")

            # 필수 필드 검증 및 기본값 설정
            validated_result = {
                "decision": result.get("decision", "HOLD").upper(),
                "percentage": min(max(float(result.get("percentage", 0)), 0), 100),  # 0-100 사이로 제한
                "reason": str(result.get("reason", "No reason provided"))
            }

            # decision 값 검증
            if validated_result["decision"] not in ["BUY", "SELL", "HOLD"]:
                validated_result["decision"] = "HOLD"

            return validated_result

        except Exception as e:
            return {
                "decision": "HOLD",
                "percentage": 0,
                "reason": f"Analysis failed: {str(e)}"
            }

    def execute_trade(self, decision: Dict) -> bool:
        """Execute trade based on analysis"""
        try:
            if decision['decision'] == 'buy':
                krw_balance = self.upbit.get_balance("KRW")
                amount = Decimal(krw_balance) * (Decimal(str(decision['percentage'])) / Decimal('100'))
                if amount > Decimal('5000'):
                    self.upbit.buy_market_order("KRW-BTC", amount * Decimal('0.9995'))
                    return True

            elif decision['decision'] == 'sell':
                btc_balance = self.upbit.get_balance("BTC")
                amount = Decimal(btc_balance) * (Decimal(str(decision['percentage'])) / Decimal('100'))
                current_price = pyupbit.get_orderbook(ticker="KRW-BTC")['orderbook_units'][0]["ask_price"]
                if Decimal(str(current_price)) * amount > Decimal('5000'):
                    self.upbit.sell_market_order("KRW-BTC", amount)
                    return True

            return False

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False


def fetch_fear_and_greed_index(limit=1, date_format=''):
    """
    Fetches the latest Fear and Greed Index data.
    Parameters:
    - limit (int): Number of results to return. Default is 1.
    - date_format (str): Date format ('us', 'cn', 'kr', 'world'). Default is '' (unixtime).
    Returns:
    - dict or str: The Fear and Greed Index data in the specified format.
    """
    base_url = "https://api.alternative.me/fng/"
    params = {
        'limit': limit,
        'format': 'json',
        'date_format': date_format
    }
    response = requests.get(base_url, params=params)
    myData = response.json()['data']
    resStr = ""
    for data in myData:
        resStr += str(data)
    return resStr

def perform_analysis():
    """Execute Bitcoin analysis and trading"""
    analyzer = BitcoinAnalyzer()

    try:
        # Gather all required data
        market_data = analyzer.get_bitcoin_data(symbol='KRW-BTC',  max_retries=3)
        if not market_data:
            print("Failed to fetch market data")
            return None
        current_price = pyupbit.get_orderbook(ticker="KRW-BTC")['orderbook_units'][0]["ask_price"]

        last_decisions = analyzer.get_last_decisions(current_price)
        fear_and_greed = fetch_fear_and_greed_index(limit=30)
        current_status = analyzer.get_current_status()
        # Generate reflection on previous trades
        reflection = analyzer.generate_trade_reflection(last_decisions, current_price)
        if not current_status:
            logger.error("Failed to get current status")
            return None

        # Analyze with GPT-4
        decision = analyzer.analyze_with_gpt4(market_data, last_decisions, fear_and_greed, current_status)

        if decision:
            # Save decision to database 결제를 해야함
            current_status_dict = json.loads(current_status)

            trading_record = TradingRecord.objects.create(
                exchange='UPBIT',
                coin_symbol='BTC',
                trade_type=decision['decision'].upper(),
                trade_ratio=Decimal(str(decision['percentage'])),
                trade_reason=decision['reason'],
                coin_balance=Decimal(current_status_dict['btc_balance']),
                balance=Decimal(current_status_dict['krw_balance']),
                current_price=Decimal(str(current_price)),
                trade_reflection=reflection
            )

            return trading_record.id

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return None






# import os
# from crewai import Agent, Task, Crew, Process
# from datetime import datetime, timedelta
# import requests
# import time
# from .models import TradingRecord
# from decimal import Decimal
#
#
# def get_current_bitcoin_price(vt_symbol):
#     try:
#         url = f"https://api.binance.com/api/v3/ticker/price?symbol={vt_symbol}"
#         response = requests.get(url)
#         data = response.json()
#         return float(data['price'])
#     except Exception as e:
#         print(f"Error fetching current Bitcoin price: {e}")
#         return None
#
#
# def get_bitcoin_data_from_api(symbol, max_retries=3):
#     base_url = "https://gridtrade.one/api/v1/binanceData/scalping"
#     session = requests.Session()
#
#     for attempt in range(max_retries):
#         try:
#             # SSL 검증을 비활성화하고 타임아웃 설정
#             response = session.get(
#                 f"{base_url}/{symbol}/1m/",
#                 timeout=30,
#                 verify=False,  # SSL 검증 비활성화
#                 headers={'User-Agent': 'Mozilla/5.0'}  # 기본 User-Agent 추가
#             )
#             response.raise_for_status()
#             data = response.json()
#
#             transformed_data = {
#                 '1m': []  # 키를 1m으로 통일
#             }
#
#             # 최근 30개의 데이터만 사용
#             recent_data = data['data'][-30:]
#
#             # 최신 지표값들 (AI 분석용)
#             latest_indicators = recent_data[-1]['indicators']
#
#             for candle in recent_data:
#                 entry = {
#                     'open': candle['open'],
#                     'high': candle['high'],
#                     'low': candle['low'],
#                     'close': candle['close'],
#                     'volume': candle['volume'],
#                     'timestamp': candle['timestamp'],
#                     'RSI': candle['indicators']['rsi'],
#                     'MACD': candle['indicators']['macd']['macd'],
#                     'MACD_Signal': candle['indicators']['macd']['signal'],
#                     'MACD_Histogram': candle['indicators']['macd']['histogram'],
#                     'MA7': candle['indicators']['moving_averages']['ma7'],
#                     'MA25': candle['indicators']['moving_averages']['ma25'],
#                     'MA99': candle['indicators']['moving_averages']['ma99'],
#                     '%K': candle['indicators']['stochastic']['k'],
#                     '%D': candle['indicators']['stochastic']['d'],
#                 }
#                 transformed_data['1m'].append(entry)
#
#             # 최신 지표값들도 포함
#             transformed_data['current_indicators'] = latest_indicators
#
#             return transformed_data
#
#         except requests.exceptions.SSLError as e:
#             print(f"SSL 오류 발생 (시도 {attempt + 1}/{max_retries}): {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(1)
#             continue
#         except requests.exceptions.RequestException as e:
#             print(f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(1)
#             continue
#         except Exception as e:
#             print(f"데이터 처리 중 오류 발생: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(1)
#             continue
#         finally:
#             session.close()
#
#     return None
#
#
# market_analyst = Agent(
#     role='Technical Market Analyst',
#     goal='Analyze market conditions and technical indicators for optimal trading decisions',
#     backstory="""You are an expert technical analyst specialized in cryptocurrency markets.
#     You excel at interpreting multiple technical indicators including RSI overbought/oversold conditions,
#     MACD momentum, and market sentiment indicators to identify high-probability trading opportunities.""",
#     verbose=True,
#     allow_delegation=False
# )
#
#
# def perform_analysis():
#     bitcoin_data = get_bitcoin_data_from_api("BTCUSDT")
#     if not bitcoin_data:
#         return None
#
#     current_indicators = bitcoin_data['current_indicators']
#     current_price = get_current_bitcoin_price("BTCUSDT")
#
#     # 기술적 지표 추출
#     current_rsi = current_indicators['rsi']
#     current_macd = current_indicators['macd']['macd']
#     current_signal = current_indicators['macd']['signal']
#     fear_greed_index = float(current_indicators['market_conditions']['fear_greed_index'])
#     price_change = float(current_indicators['market_conditions']['price_change_24h'])
#
#     # MACD 크로스오버 확인
#     prev_candle = bitcoin_data['1m'][-2]
#     prev_macd = prev_candle['MACD']
#     prev_signal = prev_candle['MACD_Signal']
#
#     macd_bullish_cross = prev_macd <= prev_signal and current_macd > current_signal
#     macd_bearish_cross = prev_macd >= prev_signal and current_macd < current_signal
#     macd_above_signal = current_macd > current_signal
#
#     # RSI 상태 판단
#     rsi_state = "neutral"
#     if current_rsi >= 70:
#         rsi_state = "overbought"
#     elif current_rsi <= 30:
#         rsi_state = "oversold"
#     elif current_rsi > 60:
#         rsi_state = "approaching_overbought"
#     elif current_rsi < 40:
#         rsi_state = "approaching_oversold"
#
#     analysis_task = Task(
#         description=f"""Analyze current market conditions with these technical indicators:
#
#         MARKET CONDITIONS:
#         - RSI: {current_rsi:.2f} (State: {rsi_state})
#         - MACD: {current_macd:.6f} (Signal: {current_signal:.6f})
#         - MACD Cross: {"Bullish" if macd_bullish_cross else "Bearish" if macd_bearish_cross else "None"}
#         - Fear & Greed Index: {fear_greed_index:.1f}
#         - Price Change 24h: {price_change:.2f}%
#
#         TRADING RULES:
#         SELL Signal (50% Position) when:
#         - RSI approaching or above 70 (overbought)
#         - MACD shows bearish momentum
#         - Extreme Greed conditions
#
#         BUY Signal (50% Position) when:
#         - RSI approaching or below 30 (oversold)
#         - MACD shows bullish momentum
#         - Extreme Fear conditions
#
#         Provide analysis in following format:
#
#         For SELL:
#         "The market is showing signs of potential overbought conditions with RSI at {current_rsi}.
#         MACD indicates [bearish momentum/divergence]. The Fear & Greed Index at {fear_greed_index:.0f}
#         suggests extreme greed. Based on these conditions, a 50% sell position is recommended."
#
#         For BUY:
#         "Market conditions show oversold signals with RSI at {current_rsi}.
#         MACD indicates [bullish momentum/convergence]. The Fear & Greed Index at {fear_greed_index:.0f}
#         suggests extreme fear. Initiating a 50% buy position."
#         """,
#         expected_output="""Please provide:
#         1. A detailed market analysis
#         2. A trading recommendation (BUY/SELL/HOLD)
#         3. Current indicator values including RSI, MACD, and Fear & Greed Index""",
#         agent=market_analyst
#     )
#
#     crew = Crew(
#         agents=[market_analyst],
#         tasks=[analysis_task],
#         verbose=True,
#         process=Process.sequential
#     )
#
#     result = crew.kickoff()
#     result_str = str(result)
#
#     try:
#         # 기본값 설정
#         action = 'HOLD'
#
#         # RSI와 MACD 조합으로 매매 결정
#         result_lower = result_str.lower()
#         if ("buy position" in result_lower and
#             (current_rsi <= 30 or rsi_state == "approaching_oversold") and
#             macd_bullish_cross):
#             action = "BUY"
#         elif ("sell position" in result_lower and
#               (current_rsi >= 70 or rsi_state == "approaching_overbought") and
#               macd_bearish_cross):
#             action = "SELL"
#
#         record = TradingRecord.objects.create(
#             timestamp=datetime.now(),
#             coin_symbol='BTCUSDT',
#             trade_type=action,
#             trade_amount_krw=Decimal('0.00'),
#             trade_reason=result_str,
#             current_price=Decimal(str(current_price)).quantize(Decimal('0.01')),
#             trade_reflection=""
#         )
#         print(f"Successfully saved trading record - Type: {action},  Price: {current_price}")
#
#     except Exception as e:
#         print(f"Error processing analysis result: {e}")
#         print(f"Result string: {result_str}")
#         return None