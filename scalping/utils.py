import os
from crewai import Agent, Task, Crew, Process
from datetime import datetime, timedelta
import requests
import time
from .models import TradingRecord
from decimal import Decimal

def get_current_bitcoin_price(vt_symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={vt_symbol}"
        response = requests.get(url)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        print(f"Error fetching current Bitcoin price: {e}")
        return None


def get_bitcoin_data_from_api(symbol, max_retries=3):
    base_url = "https://gridtrade.one/api/v1/binanceData/scalping"
    session = requests.Session()

    for attempt in range(max_retries):
        try:
            # SSL 검증을 비활성화하고 타임아웃 설정
            response = session.get(
                f"{base_url}/{symbol}/1m/",
                timeout=30,
                verify=False,  # SSL 검증 비활성화
                headers={'User-Agent': 'Mozilla/5.0'}  # 기본 User-Agent 추가
            )
            response.raise_for_status()
            data = response.json()

            transformed_data = {
                '1m': []  # 키를 1m으로 통일
            }

            # 최근 30개의 데이터만 사용
            recent_data = data['data'][-30:]

            # 최신 지표값들 (AI 분석용)
            latest_indicators = recent_data[-1]['indicators']

            for candle in recent_data:
                entry = {
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close'],
                    'volume': candle['volume'],
                    'timestamp': candle['timestamp'],
                    'RSI': candle['indicators']['rsi'],
                    'MACD': candle['indicators']['macd']['macd'],
                    'MACD_Signal': candle['indicators']['macd']['signal'],
                    'MACD_Histogram': candle['indicators']['macd']['histogram'],
                    'MA7': candle['indicators']['moving_averages']['ma7'],
                    'MA25': candle['indicators']['moving_averages']['ma25'],
                    'MA99': candle['indicators']['moving_averages']['ma99']
                }
                transformed_data['1m'].append(entry)

            # 최신 지표값들도 포함
            transformed_data['current_indicators'] = latest_indicators

            return transformed_data

        except requests.exceptions.SSLError as e:
            print(f"SSL 오류 발생 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
        except requests.exceptions.RequestException as e:
            print(f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
        except Exception as e:
            print(f"데이터 처리 중 오류 발생: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
        finally:
            session.close()

    return None


quick_analyst = Agent(
    role='Professional Scalping Trader',
    goal='Maximize profitability through precise scalping trades',
    backstory="""You are a highly skilled and experienced scalping trader known for your ability to consistently generate profits in volatile markets.
    Your deep understanding of market dynamics and technical analysis allows you to identify high-probability trading opportunities with precision.
    You excel at managing risk, setting optimal stop-loss and take-profit levels, and adapting to changing market conditions.""",
    verbose=True,
    allow_delegation=False
)


def perform_analysis():
    bitcoin_data = get_bitcoin_data_from_api("BTCUSDT")
    if not bitcoin_data:
        return None

    current_price = get_current_bitcoin_price("BTCUSDT")
    # MACD 교차 여부 확인
    macd_crossover = False
    if bitcoin_data['current_indicators']['macd']['macd'] < bitcoin_data['current_indicators']['macd']['signal']:
        macd_crossover = True


    analysis_task = Task(
        description=f"""Analyze the recent Bitcoin market data for high-probability scalping opportunities with a focus on maximizing profitability.
        Latest technical indicators:
        - RSI: {bitcoin_data['current_indicators']['rsi']} (Oversold < 30, Overbought > 70)
        - MACD: {bitcoin_data['current_indicators']['macd']['macd']}
        - MACD Signal: {bitcoin_data['current_indicators']['macd']['signal']}
        - MACD Crossover: {"Bearish" if macd_crossover else "No"}

        - Moving Averages: MA7={bitcoin_data['current_indicators']['moving_averages']['ma7']}, 
                          MA25={bitcoin_data['current_indicators']['moving_averages']['ma25']}, 
                          MA99={bitcoin_data['current_indicators']['moving_averages']['ma99']}
        - bollinger_bands : Upper={bitcoin_data['current_indicators']['bollinger_bands']['upper']},
                            Lower={bitcoin_data['current_indicators']['bollinger_bands']['lower']},
                            
        Price data for the last 30 minutes: {bitcoin_data['1m']}

        Provide a detailed trading analysis and recommendation in this exact format:

        [DECISION]
        Action: MUST be exactly BUY, SELL, or HOLD

        [ANALYSIS]
        Trade Reason: Provide a clear, concise explanation of why this trade should be executed, 
        focusing on technical indicators and market conditions. Consider the MACD crossover signal.

        [RISK MANAGEMENT]
        Stop-Loss: Specify the price level at which the trade should be closed to limit potential losses.
        Take-Profit: Define the target price for closing the trade and securing profits.

        [MARKET INSIGHT]
        Market Context: Analyze current market sentiment, trending news, and external factors that may impact the trade's success.
        Key Indicators to Monitor: Identify the most critical technical indicators to watch closely during the trade's lifecycle.
        """,
        expected_output="A precise scalping trade recommendation aimed at maximizing profitability, including entry/exit points, risk management, and market insights.",
        agent=quick_analyst
    )
    crew = Crew(
        agents=[quick_analyst],
        tasks=[analysis_task],
        verbose=True,
        process=Process.sequential
    )

    result = crew.kickoff()
    result_str = str(result)

    # Parse the AI response
    try:
        # 기본값 설정
        action = 'HOLD'
        ratio = Decimal('0.0000')
        amount = Decimal('0.00')
        reason = ""
        reflection = ""

        # 결과 파싱
        for line in result_str.split('\n'):
            line = line.strip()
            if line.startswith('Action:'):
                action = line.split('Action:')[1].strip().upper()
                if action not in ['BUY', 'SELL', 'HOLD']:
                    action = 'HOLD'  # 기본값으로 설정
            elif 'Trade Reflection:' in line:
                reflection = line.split('Trade Reflection:')[1].strip()

        # Create trading record using objects.create()
        record = TradingRecord.objects.create(
            timestamp=datetime.now(),
            coin_symbol='BTCUSDT',
            trade_type=action,
            trade_ratio=ratio,  # max_digits=5, decimal_places=4
            trade_amount_krw=amount,  # max_digits=20, decimal_places=2
            trade_reason=result_str,
            current_price=Decimal(str(current_price)).quantize(Decimal('0.01')),  # max_digits=20, decimal_places=2
            trade_reflection=reflection
        )
        print(f"Successfully saved trading record - Type: {action}, Ratio: {ratio}%, Price: {current_price}")

    except Exception as e:
        print(f"Error processing analysis result: {e}")
        print(f"Result string: {result_str}")  # 전체 결과 문자열 출력
        return None