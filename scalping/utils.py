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

    # Get all indicators from pre-calculated data
    current_indicators = bitcoin_data['current_indicators']

    # 기본 기술적 지표
    current_rsi = current_indicators['rsi']
    current_macd = current_indicators['macd']['macd']
    current_signal = current_indicators['macd']['signal']
    stoch_k = current_indicators['stochastic']['k']
    stoch_d = current_indicators['stochastic']['d']

    # 시장 상황 지표 (우리가 계산한 것들)
    fear_greed_index = float(current_indicators['market_conditions']['fear_greed_index'])
    price_change = float(current_indicators['market_conditions']['price_change_24h'])
    volatility = float(current_indicators['market_conditions']['volatility'])
    volume_ratio = float(current_indicators['market_conditions']['volume_ratio'])

    # 볼린저 밴드
    bb_upper = current_indicators['bollinger_bands']['upper']
    bb_lower = current_indicators['bollinger_bands']['lower']

    # MACD 크로스오버 확인
    prev_candle = bitcoin_data['1m'][-2]
    prev_macd = prev_candle['MACD']
    prev_signal = prev_candle['MACD_Signal']

    macd_bullish_cross = prev_macd <= prev_signal and current_macd > current_signal
    macd_bearish_cross = prev_macd >= prev_signal and current_macd < current_signal

    # MACD > Signal 여부 확인 (매수를 위한 추가 조건)
    macd_above_signal = current_macd > current_signal

    # 매매 신호 생성 - 더 엄격한 조건 적용
    is_buy_signal = (
            current_rsi <= 50 and  # RSI 조건
            macd_bullish_cross and  # MACD 골든크로스
            macd_above_signal and  # MACD가 반드시 시그널선 위에 있어야 함
            fear_greed_index < 50  # 공포 지수 조건
    )

    is_sell_signal = (
        (macd_bearish_cross and not macd_above_signal) and  # MACD 조건 (필수)
        (
            current_rsi > 50 or                # RSI 조건
            fear_greed_index >= 50            # 또는 탐욕지수 조건
        )
    )

    # MACD 상태 설정 (더 명확한 설명 추가)
    macd_state = "neutral"
    if macd_above_signal:
        macd_state = "bullish"
        macd_description = "shows upward momentum with MACD above signal line"
    else:
        macd_state = "bearish"
        macd_description = "indicates bearish momentum with MACD below signal line"

    analysis_task = Task(
        description=f"""As a professional scalping trader, analyze these technical indicators with strict MACD conditions:

        TECHNICAL CONDITIONS:
        - RSI: {current_rsi} (50 is the key level)
        - MACD: Current {current_macd}, Signal {current_signal}
        - MACD Position: {"Above Signal" if macd_above_signal else "Below Signal"} (CRITICAL for BUY signals)
        - MACD Crossover: {"Bullish" if macd_bullish_cross else "Bearish" if macd_bearish_cross else "None"}
        - Stochastic K/D: {stoch_k}/{stoch_d}
        - Fear & Greed Index: {fear_greed_index}
        - Volatility: {volatility}%
        - Volume Ratio: {volume_ratio}x

        TRADING RULES:
        SELL WHEN:
        - MACD Death Cross AND MACD below Signal line (REQUIRED)
        AND EITHER:
        - RSI > 50
        OR
        - Fear Index >= 50

        BUY WHEN:
        - RSI ≤ 50 (REQUIRED)
        AND EITHER:
        - MACD Golden Cross with MACD above Signal
        OR
        - Fear Index < 50

        Based on these conditions, provide analysis in one of these exact formats:


        For BUY Signal (ALL buy conditions must be met):
        "There's a potential upward trend for BTCUSDT. RSI {current_rsi} indicates approach to oversold territory, 
        while MACD ({macd_state}) {macd_description}. The Fear & Greed Index ({fear_greed_index:.0f}) suggests the market might be overly pessimistic. 
        The price has changed by {price_change:.2f}%. Based on this, a 50% buy position has been initiated."

        For SELL Signal:
        "Market indicators for BTCUSDT are showing a downward trend. RSI {current_rsi} suggests approach to overbought levels, 
        while MACD ({macd_state}) {macd_description}. The Fear & Greed Index ({fear_greed_index:.0f}) implies the market might be overly optimistic. 
        The price has changed by {price_change:.2f}%. Consequently, a 50% sell position has been executed."

        IMPORTANT: Never initiate a buy position when MACD is below the signal line.
        Current analysis indicates a {"BUY" if is_buy_signal else "SELL" if is_sell_signal else "HOLD"} signal.
        """,
        expected_output="A trading analysis with strict MACD conditions, especially for buy signals",
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

    try:
        # 기본값 설정
        action = 'HOLD'

        # RSI와 MACD 조합으로 매매 결정
        # AI의 분석 결과에 따라 매매 결정
        # 단순화된 결과 파싱
        result_lower = result_str.lower()
        if "buy position" in result_lower:
            action = "BUY"
        elif "sell position" in result_lower:
            action = "SELL"

        # if is_buy_signal:
        #     action = "BUY"
        # elif is_sell_signal:
        #     action = "SELL"

        record = TradingRecord.objects.create(
            timestamp=datetime.now(),
            coin_symbol='BTCUSDT',
            trade_type=action,
            trade_amount_krw=Decimal('0.00'),
            trade_reason=result_str,
            current_price=Decimal(str(current_price)).quantize(Decimal('0.01')),
            trade_reflection=""
        )
        print(f"Successfully saved trading record - Type: {action},  Price: {current_price}")

    except Exception as e:
        print(f"Error processing analysis result: {e}")
        print(f"Result string: {result_str}")
        return None
