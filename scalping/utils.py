import logging
import json
from decimal import Decimal
import requests
import pyupbit
from django.conf import settings
from openai import OpenAI
from typing import Dict, Optional, List, Optional
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
    def __init__(self, symbol):
        # self.upbit = pyupbit.Upbit(
        #     settings.UPBIT_ACCESS_KEY,
        #     settings.UPBIT_SECRET_KEY
        # )
        self.upbit = None
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.symbol = symbol
        self.base_url = f"https://gridtrade.one/api/v1/binanaceAccount"  # settings.py에 BASE_URL 추가 필요


    def _make_api_request(self, endpoint: str, method: str = 'GET', data: Dict = None) -> Dict:
        """API 요청 헬퍼 함수"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method == 'GET':
                response = requests.get(url)
            else:  # POST
                response = requests.post(url, json=data)

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_balance_from_api(self) -> Dict:
        """잔고 조회 API 호출"""
        return self._make_api_request('/upbit/balance/')

    def get_current_price_from_api(self) -> Dict:
        """현재가 조회 API 호출"""
        return self._make_api_request(f'/upbit/price/?market={self.symbol}')

    def execute_trade_via_api(self, action: str, percentage: float, is_full_trade: bool) -> Dict:
        """거래 실행 API 호출"""
        data = {
            "action": action,
            "market": self.symbol,
            "percentage": percentage,
            "is_full_trade": is_full_trade

        }
        return self._make_api_request('/upbit/trade/', method='POST', data=data)

    def get_current_status(self) -> Dict:
        """Get current trading account status using API"""
        try:
            # 현재가 조회
            price_info = self.get_current_price_from_api()
            if price_info['status'] != 'success':
                raise Exception("Failed to get current price")

            orderbook = price_info['data']
            current_time = orderbook['timestamp']

            # 잔고 조회
            balance_info = self.get_balance_from_api()
            if balance_info['status'] != 'success':
                raise Exception("Failed to get balance")

            symbol_currency = self.symbol.split('-')[1]  # KRW-BTC -> BTC
            print("★★★★★★★★★★★★★★★★★★★")
            print(balance_info)
            print("★★★★★★★★★★★★★★★★★★★")

            return json.dumps({
                'current_time': current_time,
                'orderbook': orderbook,
                f'{symbol_currency.lower()}_balance': str(
                    balance_info['data'].get(symbol_currency, {}).get('balance', '0.0')),
                'krw_balance': str(balance_info['data'].get('KRW', {}).get('balance', '0.0')),
                f'{symbol_currency.lower()}_avg_buy_price': str(
                    balance_info['data'].get(symbol_currency, {}).get('avg_buy_price', '0.0'))
            })

        except Exception as e:
            logger.error(f"Error getting current status: {e}")
            return None
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
            driver.get(f"https://upbit.com/full_chart?code=CRIX.UPBIT.{self.symbol}")

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

            # one_second_option = wait.until(
            #     EC.element_to_be_clickable((By.CSS_SELECTOR, "cq-item[stxtap*='second']"))
            # )
            # one_second_option.click()
            one_hour_option = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//cq-item[@stxtap=\"Layout.setPeriodicity(1,10,'minute')\"]")))
            one_hour_option.click()
            time.sleep(2)


            # MACD 추가
            study_menu = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "cq-menu.ciq-studies")
                )
            )
            study_menu.click()
            time.sleep(1)
            macd_item = wait.until(
                EC.presence_of_element_located((By.XPATH, "//cq-item[.//translate[@original='MACD']]"))
            )
            ActionChains(driver).move_to_element(macd_item).click().perform()
            time.sleep(1)
            ######################################

            study_menu = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "cq-menu.ciq-studies")
                )
            )
            study_menu.click()
            time.sleep(1)
            macd_item = wait.until(
                EC.presence_of_element_located((By.XPATH, "//cq-item[.//translate[@original='RSI']]"))
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

    def get_bitcoin_data(self, max_retries=3) -> Optional[Dict]:
        """Fetch Bitcoin data with technical indicators from API"""
        base_url = "https://gridtrade.one/api/v1/binanceData/upbit"
        session = requests.Session()
        for attempt in range(max_retries):
            try:
                response = session.get(
                    f"{base_url}/{self.symbol}/minute10/",
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
            - **Purpose**: This section details the insights gleaned from the most recent trading decisions undertaken by the system. It serves to provide a historical backdrop that is instrumental in refining and honing future trading strategies. Incorporate a structured evaluation of past decisions against OHLCV data to systematically assess their effectiveness.
                    - **Effectiveness**: Were recent trades profitable and aligned with market trends? 
                    - **Missed Opportunities**: Identify any missed signals or overtrading instances.
                    - **Improvements**: Suggest one quick adjustment for immediate strategy enhancement.
            - **Contents**: 
                       {previous_decisions}
                    - Each record within `last_decisions` chronicles a distinct trading decision, encapsulating the decision's timing (`timestamp`), the action executed (`decision`), the proportion of the portfolio it impacted (`percentage`), the reasoning underpinning the decision (`reason`), and the portfolio's condition at the decision's moment (`btc_balance`, `krw_balance`, `btc_avg_buy_price`).
                    - `timestamp`: Marks the exact moment the decision was recorded, expressed in milliseconds since the Unix epoch, to furnish a chronological context.
                    - `decision`: Clarifies the action taken—`buy`, `sell`, or `hold`—thus indicating the trading move made based on the analysis.
                    - `percentage`: Denotes the fraction of the portfolio allocated for the decision, mirroring the level of investment in the trading action.
                    - `reason`: Details the analytical foundation or market indicators that incited the trading decision, shedding light on the decision-making process.
                    - `btc_balance`: Reveals the quantity of Bitcoin within the portfolio at the decision's time, demonstrating the portfolio's market exposure.
                    - `krw_balance`: Indicates the amount of Korean Won available for trading at the time of the decision, signaling liquidity.
                    - `btc_avg_buy_price`: Provides the average acquisition cost of the Bitcoin holdings, serving as a metric for evaluating the past decisions' performance and the prospective future profitability.
                    - `current_price`: {current_price}
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a cryptocurrency trading advisor providing reflections on past trades."},
                    {"role": "user", "content": reflection_prompt}
                ],
                max_tokens=200  # 짧게 유지
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            return "반성 생성 중 오류 발생"

    def execute_trade(self, decision: Dict) -> bool:
        """Execute trade based on analysis"""
        try:
            if decision['decision'] == 'HOLD':
                logger.info("Decision is HOLD - no trade executed")
                return True

            if decision['decision'] not in ['BUY', 'SELL']:
                return False

            # 마지막 매도 시점 찾기
            latest_sell = TradingRecord.objects.filter(
                exchange='UPBIT',
                coin_symbol=self.symbol.split('-')[1],
                trade_type='SELL'
            ).order_by('-created_at').first()

            # 마지막 매도 이후의 거래만 체크 (HOLD 제외)
            if latest_sell:
                current_cycle_trades = TradingRecord.objects.filter(
                    exchange='UPBIT',
                    coin_symbol=self.symbol.split('-')[1],
                    created_at__gt=latest_sell.created_at
                ).exclude(trade_type='HOLD')  # HOLD 제외
            else:
                # 매도가 없으면 모든 거래 체크 (HOLD 제외)
                current_cycle_trades = TradingRecord.objects.filter(
                    exchange='UPBIT',
                    coin_symbol=self.symbol.split('-')[1]
                ).exclude(trade_type='HOLD')  # HOLD 제외

            # 현재 사이클의 매수/매도 횟수
            current_buy_count = current_cycle_trades.filter(trade_type='BUY').count()
            current_sell_count = current_cycle_trades.filter(trade_type='SELL').count()

            if decision['decision'] == 'BUY':
                if current_buy_count == 0:
                    # 첫 매수는 50%
                    is_full_trade = False
                elif current_buy_count == 1:
                    # 두 번째 매수는 전체
                    is_full_trade = True
                else:
                    logger.info("Already bought twice in current cycle")
                    return False

            else:  # SELL
                if current_buy_count == 0:
                    logger.info("No buy records found in current cycle")
                    return False
                elif current_buy_count == 1:
                    # 매수가 1번이면 전체 매도
                    is_full_trade = True
                else:  # current_buy_count == 2
                    if current_sell_count == 0:
                        # 첫 매도는 50%
                        is_full_trade = False
                    else:
                        # 두 번째 매도는 전체
                        is_full_trade = True

            # 거래 실행
            trade_result = self.execute_trade_via_api(
                action=decision['decision'].lower(),
                percentage=float(decision['percentage']),
                is_full_trade=is_full_trade
            )

            return trade_result['status'] == 'success'

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False
    # percentage = float(decision['percentage']),

    def get_last_decisions(self, num_decisions: int = 5, current_price = 100000000) -> str:
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
                    "current_price": {current_price},
                    "avg_buy_price": float(decision.avg_buy_price)  # btc_avg_buy_price -> avg_buy_price로 수정
                }
                formatted_decisions.append(str(formatted_decision))

            return "\n".join(formatted_decisions)
        except Exception as e:
            logger.error(f"Error getting last decisions: {e}")
            return ""

    def _format_last_row_indicators(self, market_data: Dict) -> str:
        """Format last row's technical indicators into a readable string"""
        try:
            columns = market_data['columns']
            last_row = market_data['data'][-1]

            # 주요 지표들과 현재 가격 정보
            formatted_indicators = [
                "=== Current Market Indicators ===",
                f"Price: {last_row[columns.index('close'):]}",
                f"Volume: {last_row[columns.index('volume')]}",
                "",
                "=== Technical Indicators ===",
                f"RSI(14): {last_row[columns.index('RSI_14')]:.2f}",
                f"MACD: {last_row[columns.index('MACD')]:.2f}",
                f"MACD Signal: {last_row[columns.index('Signal_Line')]:.2f}",
                f"MACD Histogram: {last_row[columns.index('MACD_Histogram')]:.2f}",
                "",
                "=== Moving Averages ===",
                f"MA7: {last_row[columns.index('MA7')]:.2f}",
                f"MA25: {last_row[columns.index('MA25')]:.2f}",
                f"MA99: {last_row[columns.index('MA99')]:.2f}",
                "",
                "=== Bollinger Bands ===",
                f"Upper Band: {last_row[columns.index('Upper_Band')]:.2f}",
                f"Middle Band: {last_row[columns.index('Middle_Band')]:.2f}",
                f"Lower Band: {last_row[columns.index('Lower_Band')]:.2f}",
                "",
            ]

            return "\n".join(formatted_indicators)
        except Exception as e:
            logger.error(f"Error formatting indicators: {e}")
            return "Error occurred while formatting market indicators"

    def analyze_with_gpt4(self, market_data: Dict, reflection, fear_and_greed, current_status: str) -> Optional[Dict]:
        """Analyze market data using GPT-4"""
        try:
            # 현재 파일의 디렉토리 경로를 가져옴
            import os

            current_dir = os.path.dirname(os.path.abspath(__file__))
            instructions_path = os.path.join(current_dir, 'instructions_v3.md')

            with open(instructions_path, 'r', encoding='utf-8') as file:
                instructions = file.read()


            current_indicators = self._format_last_row_indicators(market_data)
            # GPT에 보낼 프롬프트 구성
            analysis_prompt = f"""
            Current Technical Indicators:
            {current_indicators}
            """

            # 차트 이미지 캡처
            chart_image = self.capture_chart()
            print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
            print(current_indicators)
            print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")

            if not chart_image:
                logger.warning("Failed to capture chart image")
                # {"role": "user", "content": fear_and_greed},
                # {"role": "user", "content": analysis_prompt},

            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(market_data)},
                {"role": "user", "content": reflection},
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
                model="gpt-4o",  # 이미지를 처리할 수 있는 모델로 변경
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=800
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

    # def execute_trade(self, decision: Dict) -> bool:
    #     """Execute trade based on analysis"""
    #     try:
    #         if decision['decision'] == 'buy':
    #             krw_balance = self.upbit.get_balance("KRW")
    #             amount = Decimal(krw_balance) * (Decimal(str(decision['percentage'])) / Decimal('100'))
    #             if amount > Decimal('5000'):
    #                 self.upbit.buy_market_order("KRW-BTC", amount * Decimal('0.9995'))
    #                 return True
    #
    #         elif decision['decision'] == 'sell':
    #             btc_balance = self.upbit.get_balance("BTC")
    #             amount = Decimal(btc_balance) * (Decimal(str(decision['percentage'])) / Decimal('100'))
    #             current_price = pyupbit.get_orderbook(ticker="KRW-BTC")['orderbook_units'][0]["ask_price"]
    #             if Decimal(str(current_price)) * amount > Decimal('5000'):
    #                 self.upbit.sell_market_order("KRW-BTC", amount)
    #                 return True
    #
    #         return False
    #
    #     except Exception as e:
    #         logger.error(f"Trade execution error: {e}")
    #         return False


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


# def perform_analysis(symbol):
#     """Execute Bitcoin analysis and trading"""
#     analyzer = BitcoinAnalyzer(symbol)
#
#     try:
#         # 가장 최근 거래 기록 확인
#         try:
#             latest_record = TradingRecord.objects.filter(
#                 exchange='UPBIT',
#                 coin_symbol=symbol.split('-')[1]
#             ).latest('created_at')
#             last_trade_type = latest_record.trade_type
#         except TradingRecord.DoesNotExist:
#             last_trade_type = None
#
#         # Gather all required data
#         market_data = analyzer.get_bitcoin_data(max_retries=3)
#         if not market_data:
#             logger.error("Failed to fetch market data")
#             return None
#
#         current_price = pyupbit.get_orderbook(ticker=symbol)['orderbook_units'][0]["ask_price"]
#         last_decisions = analyzer.get_last_decisions(current_price)
#         fear_and_greed = fetch_fear_and_greed_index(limit=30)
#         current_status = analyzer.get_current_status()
#         reflection = analyzer.generate_trade_reflection(last_decisions, current_price)
#
#         if not current_status:
#             logger.error("Failed to get current status")
#             return None
#
#         # Analyze with GPT-4
#         decision = analyzer.analyze_with_gpt4(
#             market_data,
#             last_decisions,
#             fear_and_greed,
#             current_status
#         )
#
#         if decision:
#             # Save decision to database 결제를 해야함
#             current_status_dict = json.loads(current_status)
#
#             trading_record = TradingRecord.objects.create(
#                 exchange='UPBIT',
#                 coin_symbol=symbol.split('-')[1],  # 심볼에서 'BTC' 부분만 추출
#                 trade_type=decision['decision'].upper(),
#                 trade_ratio=Decimal(str(decision['percentage'])),
#                 trade_reason=decision['reason'],
#                 coin_balance=Decimal(current_status_dict['btc_balance']),
#                 balance=Decimal(current_status_dict['krw_balance']),
#                 current_price=Decimal(str(current_price)),
#                 trade_reflection=reflection
#             )
#
#             return trading_record.id
#         return None
#
#     except Exception as e:
#         logger.error(f"Analysis error: {e}")
#         return None

def perform_analysis(symbol):
    """Execute Bitcoin analysis and trading"""
    analyzer = BitcoinAnalyzer(symbol)

    try:
        # 가장 최근 거래 기록 확인
        try:
            latest_record = TradingRecord.objects.filter(
                exchange='UPBIT',
                coin_symbol=symbol.split('-')[1]
            ).latest('created_at')
            last_trade_type = latest_record.trade_type
            coin_balance = latest_record.coin_balance
        except TradingRecord.DoesNotExist:
            last_trade_type = None
            coin_balance = Decimal('0')

        # 현재가 조회
        current_price = pyupbit.get_orderbook(ticker=symbol)['orderbook_units'][0]["ask_price"]

        # 보유 코인의 현재 가치 계산
        coin_value_krw = coin_balance * Decimal(str(current_price))

        # 현재 포지션 상태 파악 (5000원 기준)
        allowed_trade_type = None
        if coin_value_krw >= Decimal('5000'):
            # 의미 있는 코인 보유량이 있으면 매도만 가능
            allowed_trade_type = 'SELL'
            logger.info(f"Current coin value: {coin_value_krw} KRW - Position: HOLD")
        else:
            # 의미 있는 코인 보유량이 없으면 매수만 가능
            allowed_trade_type = 'BUY'
            logger.info(f"Current coin value: {coin_value_krw} KRW - Position: READY")

        # 데이터 수집
        market_data = analyzer.get_bitcoin_data(max_retries=3)
        if not market_data:
            logger.error("Failed to fetch market data")
            return None

        last_decisions = analyzer.get_last_decisions(current_price)
        fear_and_greed = fetch_fear_and_greed_index(limit=30)
        reflection = analyzer.generate_trade_reflection(last_decisions, current_price)
        current_status = analyzer.get_current_status()
        current_status_dict = json.loads(current_status)

        # 평균매수가 추출
        symbol_currency = symbol.split('-')[1]  # KRW-BTC -> BTC
        avg_buy_price = float(current_status_dict.get(f'{symbol_currency.lower()}_avg_buy_price', '0.0'))



        if not current_status:
            logger.error("Failed to get current status")
            return None

        # GPT-4 분석
        decision = analyzer.analyze_with_gpt4(
            market_data,
            last_decisions,
            fear_and_greed,
            current_status
        )

        if not decision:
            logger.error("Failed to get analysis decision")
            return None

        should_execute = True

        # if decision['decision'] == 'HOLD':
        #     should_execute = False
        # elif decision['decision'] == last_trade_type:
        #     logger.info(f"Skipping {decision['decision']} - Same as last trade type")
        #     should_execute = False
        # elif decision['decision'] == 'BUY':
        #     # 매수 시에는 KRW 잔고만 확인
        #     krw_balance = Decimal(current_status_dict['krw_balance'])
        #     trade_amount = krw_balance
        #     if trade_amount < Decimal('5000'):
        #         logger.info(f"Skipping BUY - Trade amount ({trade_amount} KRW) is less than minimum (5000 KRW)")
        #         should_execute = False

        # 실제 거래 실행
        if should_execute:
            trade_success = analyzer.execute_trade(decision)
            if not trade_success:
                logger.error("Trade execution failed")


        trading_record = TradingRecord.objects.create(
            exchange='UPBIT',
            coin_symbol=symbol.split('-')[1],
            trade_type=decision['decision'].upper(),
            trade_ratio=Decimal(str(decision['percentage'])),
            trade_reason=decision['reason'],
            coin_balance=Decimal(current_status_dict[f'{symbol.split("-")[1].lower()}_balance']),
            balance=Decimal(current_status_dict['krw_balance']),
            current_price=Decimal(str(current_price)),
            trade_reflection=reflection,
            avg_buy_price=Decimal(str(avg_buy_price)),  # btc_avg_buy_price -> avg_buy_price로 수정
        )
        # 거래 실행 조건 확인


        return trading_record.id
        return None

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return None

