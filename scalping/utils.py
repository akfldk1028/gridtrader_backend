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


market_analyst = Agent(
    role='Technical Market Analyst',
    goal='Analyze market conditions and technical indicators for optimal trading decisions',
    backstory="""You are an expert technical analyst specialized in cryptocurrency markets.
    You excel at interpreting multiple technical indicators including RSI overbought/oversold conditions,
    MACD momentum, and market sentiment indicators to identify high-probability trading opportunities.""",
    verbose=True,
    allow_delegation=False
)


def perform_analysis():
    bitcoin_data = get_bitcoin_data_from_api("BTCUSDT")
    if not bitcoin_data:
        return None

    current_indicators = bitcoin_data['current_indicators']
    current_price = get_current_bitcoin_price("BTCUSDT")


    # 기술적 지표 추출
    current_rsi = current_indicators['rsi']
    current_macd = current_indicators['macd']['macd']
    current_signal = current_indicators['macd']['signal']
    fear_greed_index = float(current_indicators['market_conditions']['fear_greed_index'])
    price_change = float(current_indicators['market_conditions']['price_change_24h'])

    # MACD 크로스오버 확인
    prev_candle = bitcoin_data['1m'][-2]
    prev_macd = prev_candle['MACD']
    prev_signal = prev_candle['MACD_Signal']

    macd_bullish_cross = prev_macd <= prev_signal and current_macd > current_signal
    macd_bearish_cross = prev_macd >= prev_signal and current_macd < current_signal
    macd_above_signal = current_macd > current_signal

    # RSI 상태 판단
    rsi_state = "neutral"
    if current_rsi >= 70:
        rsi_state = "overbought"
    elif current_rsi <= 30:
        rsi_state = "oversold"
    elif current_rsi > 60:
        rsi_state = "approaching_overbought"
    elif current_rsi < 40:
        rsi_state = "approaching_oversold"


    analysis_task = Task(
        description=f"""Analyze current market conditions with these technical indicators:

        MARKET CONDITIONS:
        - RSI: {current_rsi:.2f} (State: {rsi_state})
        - MACD: {current_macd:.6f} (Signal: {current_signal:.6f})
        - MACD Cross: {"Bullish" if macd_bullish_cross else "Bearish" if macd_bearish_cross else "None"}
        - Fear & Greed Index: {fear_greed_index:.1f}
        - Price Change 24h: {price_change:.2f}%


        TRADING RULES:
        SELL Signal (50% Position) when:
        - RSI approaching or above 70 (overbought)
        - MACD shows bearish momentum
        - Extreme Greed conditions

        BUY Signal (50% Position) when:
        - RSI approaching or below 30 (oversold)
        - MACD shows bullish momentum
        - Extreme Fear conditions

        Provide analysis in following format:


        For SELL:
        "The market is showing signs of potential overbought conditions with RSI at {current_rsi}. 
        MACD indicates [bearish momentum/divergence]. The Fear & Greed Index at {fear_greed_index:.0f} 
        suggests extreme greed. Based on these conditions, a 50% sell position is recommended."

        For BUY:
        "Market conditions show oversold signals with RSI at {current_rsi}. 
        MACD indicates [bullish momentum/convergence]. The Fear & Greed Index at {fear_greed_index:.0f} 
        suggests extreme fear. Initiating a 50% buy position."
        """,
        agent=market_analyst
    )

    crew = Crew(
        agents=[market_analyst],
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
        if ("buy position" in result_lower and
            (current_rsi <= 30 or rsi_state == "approaching_oversold") and
            macd_bullish_cross):
            action = "BUY"
        elif ("sell position" in result_lower and
              (current_rsi >= 70 or rsi_state == "approaching_overbought") and
              macd_bearish_cross):
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
