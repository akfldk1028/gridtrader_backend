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

    for attempt in range(max_retries):
        try:
            response = requests.get(f"{base_url}/{symbol}/1m/", timeout=30)
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
                    'MA5': candle['indicators']['moving_averages']['ma5'],
                    'MA10': candle['indicators']['moving_averages']['ma10'],
                    'MA20': candle['indicators']['moving_averages']['ma20']
                }
                transformed_data['1m'].append(entry)  # 1m으로 수정

            # 최신 지표값들도 포함
            transformed_data['current_indicators'] = latest_indicators

            return transformed_data
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

    return None


quick_analyst = Agent(
    role='Scalping Analyst',
    goal='Provide quick trading signals for short-term cryptocurrency trading',
    backstory="""You are an experienced scalping trader who specializes in quick market analysis 
    and providing actionable trading signals. You focus on immediate market conditions and technical indicators 
    to make rapid trading decisions.""",
    verbose=True,
    allow_delegation=False
)


def perform_analysis():
    bitcoin_data = get_bitcoin_data_from_api("BTCUSDT")
    if not bitcoin_data:
        return None

    current_price = get_current_bitcoin_price("BTCUSDT")

    analysis_task = Task(
        description=f"""Analyze the recent Bitcoin market data for scalping opportunities.
        Latest technical indicators:
        - RSI: {bitcoin_data['current_indicators']['rsi']} (Oversold < 30, Overbought > 70)
        - MACD: {bitcoin_data['current_indicators']['macd']['macd']}
        - MACD Signal: {bitcoin_data['current_indicators']['macd']['signal']}
        - MACD Histogram: {bitcoin_data['current_indicators']['macd']['histogram']}
        - Moving Averages: MA5={bitcoin_data['current_indicators']['moving_averages']['ma5']}, 
                          MA10={bitcoin_data['current_indicators']['moving_averages']['ma10']}, 
                          MA20={bitcoin_data['current_indicators']['moving_averages']['ma20']}

        Price data for the last 30 minutes: {bitcoin_data['1m']}

        Provide a detailed trading analysis and recommendation in this exact format:

        [DECISION]
        Action: MUST be exactly BUY, SELL, or HOLD
        Trading Ratio: Specify a precise percentage between 0-100% with exactly 4 decimal places
        Amount KRW: Suggested amount in KRW (Korean Won)

        [ANALYSIS]
        Trade Reason: Provide a clear, concise explanation of why this trade should be executed, 
        focusing on technical indicators and market conditions.

        [RISK ANALYSIS]
        Trade Reflection: Analyze the following points:
        1. Key Risk Factors: What could make this trade unsuccessful?
        2. Counter Signals: Are there any indicators suggesting the opposite direction?
        3. Risk Management: Suggested stop-loss and take-profit levels
        4. Market Context: Current market sentiment and external factors to watch
        5. Key Indicators to Monitor: Which technical indicators should be closely watched after entering this position

        Note: All numerical values must be precise and formatted exactly as specified.
        Current price: {current_price} KRW
        """,
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
            elif line.startswith('Trading Ratio:'):
                ratio = Decimal(line.split('Trading Ratio:')[1].strip().rstrip('%'))
            elif line.startswith('Amount KRW:'):
                amount = Decimal(line.split('Amount KRW:')[1].strip().replace(',', ''))
            elif 'Trade Reason:' in line:
                reason = line.split('Trade Reason:')[1].strip()
            elif 'Trade Reflection:' in line:
                reflection = line.split('Trade Reflection:')[1].strip()

        # Create trading record object
        record = TradingRecord(
            timestamp=datetime.now(),
            coin_symbol='BTCUSDT',
            trade_type=action,
            trade_ratio=ratio,
            trade_amount_krw=amount,
            trade_reason=reason,
            coin_balance=Decimal('0.5000'),  # 예시값, 실제로는 현재 잔고를 가져와야 함
            balance=Decimal('10000000.00'),  # 예시값, 실제로는 현재 잔고를 가져와야 함
            current_price=Decimal(str(current_price)),
            trade_reflection=reflection
        )

        # Save the record
        record.save()



    except Exception as e:
        print(f"Error processing analysis result: {e}")
        return None