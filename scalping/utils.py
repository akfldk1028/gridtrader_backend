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
                    'MA99': candle['indicators']['moving_averages']['ma99'],
                    '%K': candle['indicators']['stochastic']['k'],
                    '%D': candle['indicators']['stochastic']['d'],
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

    # Get all pre-calculated indicators including fear & greed index
    current_indicators = bitcoin_data['current_indicators']

    fear_greed_index = current_indicators['market_conditions']['fear_greed_index']
    price_change_24h = current_indicators['market_conditions']['price_change_24h']
    volatility = current_indicators['market_conditions']['volatility']
    volume_ratio = current_indicators['market_conditions']['volume_ratio']

    # Get other technical indicators
    current_macd = current_indicators['macd']['macd']
    current_signal = current_indicators['macd']['signal']
    current_rsi = current_indicators['rsi']
    stoch_k = current_indicators['stochastic']['k']
    stoch_d = current_indicators['stochastic']['d']
    bb_upper = current_indicators['bollinger_bands']['upper']
    bb_lower = current_indicators['bollinger_bands']['lower']

    # Determine market sentiment based on actual fear & greed index
    if fear_greed_index < 50:
        market_sentiment = "suggests the market might be overly pessimistic"
    else:
        market_sentiment = "implies the market might be overly optimistic"

    # Determine MACD state
    macd_state = "neutral"
    if current_macd > current_signal:
        macd_state = "shows upward momentum"
    else:
        macd_state = "indicates a bearish crossover"

    analysis_task = Task(
        description=f"""As a professional scalping trader, analyze these comprehensive market conditions:

        TECHNICAL INDICATORS:
        - RSI: {current_rsi}
        - MACD: {current_macd} (Signal: {current_signal})
        - Stochastic K/D: {stoch_k}/{stoch_d}
        - Bollinger Bands: Upper {bb_upper}, Lower {bb_lower}

        MARKET CONDITIONS:
        - Fear & Greed Index: {fear_greed_index}
        - 24h Price Change: {price_change_24h}%
        - Volatility: {volatility}%
        - Volume Ratio: {volume_ratio}x

        Provide your analysis in exactly one of these formats:

        For BUY signals:
        "There's a potential upward trend for BTCUSDT. RSI ({current_rsi:.0f}) indicates approach to oversold territory, 
        while MACD ({macd_state}) {macd_state}. The Fear & Greed Index ({fear_greed_index:.0f}) {market_sentiment}. 
        The price has changed by {price_change_24h:.2f}%. Based on this, a 50% buy position has been initiated."

        For SELL signals:
        "Market indicators for BTCUSDT are showing a downward trend. RSI ({current_rsi:.0f}) suggests approach to overbought levels, 
        while MACD ({macd_state}) {macd_state}. The Fear & Greed Index ({fear_greed_index:.0f}) {market_sentiment}. 
        The price has changed by {price_change_24h:.2f}%. Consequently, a 50% sell position has been executed."

        Choose the appropriate format based on technical indicators and market conditions.
        """,
        expected_output="A detailed trading analysis using all available technical indicators",
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
        if "buy position has been initiated" in result_str:
            action = "BUY"
        elif "sell position has been executed" in result_str:
            action = "SELL"

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
