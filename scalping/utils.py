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

    def execute_trade_via_api(self, action: str, percentage: float) -> Dict:
        """거래 실행 API 호출"""
        data = {
            "action": action,
            "market": self.symbol,
            "percentage": percentage
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

    def get_bitcoin_data(self, max_retries=3) -> Optional[Dict]:
        """Fetch Bitcoin data with technical indicators from API"""
        base_url = "https://gridtrade.one/api/v1/binanceData/upbit"
        session = requests.Session()
        for attempt in range(max_retries):
            try:
                response = session.get(
                    f"{base_url}/{self.symbol}/minute1/",
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
    # def get_current_status(self) -> Dict:
    #     """Get current trading account status"""
    #     try:
    #         orderbook = pyupbit.get_orderbook(ticker=self.symbol)
    #         current_time = orderbook['timestamp']
    #
    #         # Upbit 연동이 있는 경우 실제 데이터 조회
    #         balances = self.upbit.get_balances()
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

    def execute_trade(self, decision: Dict) -> bool:
        """Execute trade based on analysis using API"""
        try:
            if decision['decision'] not in ['BUY', 'SELL']:
                return False

            trade_result = self.execute_trade_via_api(
                action=decision['decision'].lower(),
                percentage=float(decision['percentage'])
            )

            return trade_result['status'] == 'success'

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False

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
        except TradingRecord.DoesNotExist:
            last_trade_type = None

        # Gather all required data
        market_data = analyzer.get_bitcoin_data(max_retries=3)
        if not market_data:
            logger.error("Failed to fetch market data")
            return None

        current_price = pyupbit.get_orderbook(ticker=symbol)['orderbook_units'][0]["ask_price"]
        last_decisions = analyzer.get_last_decisions(current_price)
        fear_and_greed = fetch_fear_and_greed_index(limit=30)
        current_status = analyzer.get_current_status()
        reflection = analyzer.generate_trade_reflection(last_decisions, current_price)

        if not current_status:
            logger.error("Failed to get current status")
            return None

        # Analyze with GPT-4
        decision = analyzer.analyze_with_gpt4(
            market_data,
            last_decisions,
            fear_and_greed,
            current_status
        )

        if decision:
            # Save decision to database 결제를 해야함
            current_status_dict = json.loads(current_status)

            trading_record = TradingRecord.objects.create(
                exchange='UPBIT',
                coin_symbol=symbol.split('-')[1],  # 심볼에서 'BTC' 부분만 추출
                trade_type=decision['decision'].upper(),
                trade_ratio=Decimal(str(decision['percentage'])),
                trade_reason=decision['reason'],
                coin_balance=Decimal(current_status_dict['btc_balance']),
                balance=Decimal(current_status_dict['krw_balance']),
                current_price=Decimal(str(current_price)),
                trade_reflection=reflection
            )

            return trading_record.id
        # if not decision:
        #     logger.error("Failed to get GPT-4 analysis")
        #     return None
        #
        # # 거래 시그널 검증
        # should_execute = True
        # should_record = True
        #
        # if decision['decision'] == 'HOLD':
        #     should_execute = False
        #     should_record = True  # HOLD도 기록은 남김
        #
        # elif decision['decision'] == last_trade_type:
        #     # 직전 거래와 동일한 타입인 경우 거래와 기록 모두 스킵
        #     logger.info(f"Skipping {decision['decision']} - Same as last trade type")
        #     return None
        #
        # if should_execute or should_record:
        #     current_status_dict = json.loads(current_status)
        #
        #     # 실제 거래 실행
        #     if should_execute:
        #         trade_success = analyzer.execute_trade(decision)
        #         if not trade_success:
        #             logger.error("Trade execution failed")
        #             # 거래는 실패해도 기록은 남김
        #
        #     # Generate reflection
        #     reflection = analyzer.generate_trade_reflection(last_decisions, current_price)
        #
        #     # Save decision to database
        #     trading_record = TradingRecord.objects.create(
        #         exchange='UPBIT',
        #         coin_symbol=symbol.split('-')[1],  # 심볼에서 'BTC' 부분만 추출
        #         trade_type=decision['decision'].upper(),
        #         trade_ratio=Decimal(str(decision['percentage'])),
        #         trade_reason=decision['reason'],
        #         coin_balance=Decimal(current_status_dict['btc_balance']),
        #         balance=Decimal(current_status_dict['krw_balance']),
        #         current_price=Decimal(str(current_price)),
        #         trade_reflection=reflection
        #     )
        #
        #     return trading_record.id

        return None

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return None

# def perform_analysis(symbol='KRW-BTC'):
#     """Execute Bitcoin analysis and trading"""
#     analyzer = BitcoinAnalyzer(symbol)
#
#     try:
#         # Gather all required data
#         market_data = analyzer.get_bitcoin_data(max_retries=3)
#         if not market_data:
#             print("Failed to fetch market data")
#             return None
#         current_price = pyupbit.get_orderbook(ticker=symbol)['orderbook_units'][0]["ask_price"]
#
#         last_decisions = analyzer.get_last_decisions(current_price)
#         fear_and_greed = fetch_fear_and_greed_index(limit=30)
#         current_status = analyzer.get_current_status()
#         # Generate reflection on previous trades
#         reflection = analyzer.generate_trade_reflection(last_decisions, current_price)
#         if not current_status:
#             logger.error("Failed to get current status")
#             return None
#
#         # Analyze with GPT-4
#         decision = analyzer.analyze_with_gpt4(market_data, last_decisions, fear_and_greed, current_status)
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
#
#     except Exception as e:
#         logger.error(f"Analysis error: {e}")
#         return None


class MultiCoinAnalyzer:
    def __init__(self, symbols: List[str] = None, total_balance: Decimal = Decimal('3000000')):
        self.symbols = symbols or ['KRW-BTC', 'KRW-ETH', 'KRW-XRP']
        self.total_balance = total_balance
        self.per_symbol_balance = total_balance / len(self.symbols)
        self.analyzers = {
            symbol: BitcoinAnalyzer(symbol=symbol)
            for symbol in self.symbols
        }

    def perform_multi_analysis(self) -> List[Dict]:
        """모든 심볼에 대한 분석 수행"""
        results = []

        for symbol in self.symbols:
            try:
                analyzer = self.analyzers[symbol]

                # 기존 BitcoinAnalyzer의 분석 로직 실행
                market_data = analyzer.get_bitcoin_data(max_retries=3)
                if not market_data:
                    logger.error(f"Failed to fetch market data for {symbol}")
                    continue

                current_price = pyupbit.get_orderbook(ticker=symbol)['orderbook_units'][0]["ask_price"]
                current_status = analyzer.get_current_status()

                if not current_status:
                    logger.error(f"Failed to get current status for {symbol}")
                    continue

                # 해당 심볌의 이전 거래 내역 조회
                last_decisions = TradingRecord.objects.filter(
                    exchange='UPBIT',
                    coin_symbol=symbol.split('-')[1]
                ).order_by('-created_at')[:10]

                # 거래 내역 포맷팅
                formatted_decisions = []
                for decision in last_decisions:
                    formatted_decision = {
                        "timestamp": int(decision.created_at.timestamp() * 1000),
                        "decision": decision.trade_type.lower(),
                        "percentage": float(decision.trade_ratio),
                        "reason": decision.trade_reason,
                        "coin_balance": float(decision.coin_balance),
                        "krw_balance": float(decision.balance),
                        "current_price": float(decision.current_price)
                    }
                    formatted_decisions.append(str(formatted_decision))

                last_decisions_str = "\n".join(formatted_decisions)

                # 기술적 분석 데이터 수집
                fear_and_greed = fetch_fear_and_greed_index(limit=30)

                # GPT 분석 실행
                decision = analyzer.analyze_with_gpt4(
                    market_data,
                    last_decisions_str,
                    fear_and_greed,
                    current_status
                )

                if not decision:
                    logger.error(f"Analysis failed for {symbol}")
                    continue

                if decision['decision'] != 'HOLD':
                    # 거래 전 현재 잔고 확인
                    try:
                        latest_record = TradingRecord.objects.filter(
                            exchange='UPBIT',
                            coin_symbol=symbol.split('-')[1]
                        ).latest('created_at')
                        krw_balance = latest_record.balance
                        coin_balance = latest_record.coin_balance
                    except TradingRecord.DoesNotExist:
                        krw_balance = self.per_symbol_balance
                        coin_balance = Decimal('0')

                    # 거래 금액 계산
                    trade_ratio = Decimal(str(decision['percentage'])) / Decimal('100')

                    if decision['decision'].upper() == 'BUY':
                        trade_amount_krw = krw_balance * trade_ratio
                        trade_coin_amount = trade_amount_krw / Decimal(str(current_price))
                        new_krw_balance = krw_balance - trade_amount_krw
                        new_coin_balance = coin_balance + trade_coin_amount
                    else:  # SELL
                        trade_coin_amount = coin_balance * trade_ratio
                        trade_amount_krw = trade_coin_amount * Decimal(str(current_price))
                        new_krw_balance = krw_balance + trade_amount_krw
                        new_coin_balance = coin_balance - trade_coin_amount

                    # 거래 기록 저장
                    trading_record = TradingRecord.objects.create(
                        exchange='UPBIT',
                        coin_symbol=symbol.split('-')[1],
                        trade_type=decision['decision'].upper(),
                        trade_amount_krw=trade_amount_krw,
                        trade_ratio=decision['percentage'],
                        coin_balance=new_coin_balance,
                        balance=new_krw_balance,
                        current_price=Decimal(str(current_price)),
                        trade_reason=decision['reason'],
                        trade_reflection=analyzer.generate_trade_reflection(
                            last_decisions_str,
                            Decimal(str(current_price))
                        )
                    )

                    results.append({
                        'symbol': symbol,
                        'record_id': trading_record.id,
                        'decision': decision,
                        'trade_amount': float(trade_amount_krw)
                    })

            except Exception as e:
                logger.error(f"Error in analysis for {symbol}: {e}")
                continue

        return results


