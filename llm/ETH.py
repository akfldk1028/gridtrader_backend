import os
from crewai import Agent, Task, Crew, Process
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np
import re
from datetime import datetime
from django.conf import settings
from TradeStrategy.models import StrategyConfig
# from common.models import CommonModel
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from .analysis.StochasticRSI import StochasticRSI
from .analysis.rsi import RSIAnalyzer
from datetime import datetime, timedelta
import requests
from rest_framework.test import APIRequestFactory
from binanceData.views import BinanceLLMChartDataAPIView
from rest_framework.response import Response

# Binance 클라이언트 초기화
symbol = "ETHUSDT"

# binance_api_key = settings.BINANCE_API_KEY
# binance_api_secret = settings.BINANCE_API_SECRET
# client = Client(binance_api_key, binance_api_secret)
cryptocompare_api_key = "400daae3cf09044e5d78b3fc744b107731547031372de5573431166b96d16db7"


def get_crypto_news():
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={cryptocompare_api_key}"
    response = requests.get(url)
    news_data = response.json()['Data']
    return news_data[:30]  # 최근 10개의 뉴스만 반환


news_analyst = Agent(
    role='Crypto News Trend Analyst',
    goal='Analyze recent cryptocurrency news to predict short-term and long-term Bitcoin price trends',
    backstory="""You are a seasoned crypto news analyst with an exceptional ability to predict market trends. 
    Your expertise lies in quickly digesting news information and translating it into actionable trend forecasts for Bitcoin. 
    You have a proven track record of accurately predicting both short-term (12-24 hours) and long-term (1-2 days) price movements based on news sentiment and market-moving events.""",
    verbose=True,
    allow_delegation=False,

)



fifteen_min_analyst = Agent(
    role=f'15-Minute Market Analyst',
    goal='Analyze Bitcoin market trends and patterns in 15-minute timeframe, with emphasis on Ichimoku Cloud indicators',
    backstory="""You are an experienced cryptocurrency market analyst specializing in short-term Bitcoin analysis.
    Your expertise lies in technical analysis and identifying market trends in 15-minute charts.
    You are particularly skilled in interpreting Ichimoku Cloud indicators to predict market movements.
   You are known for your quick analysis and ability to spot rapid market changes.""",
    verbose=True,
    allow_delegation=False,
)

thirty_min_analyst = Agent(
    role=f'30-Minute Market Analyst',
    goal='Analyze Bitcoin market trends and patterns in 30-minute timeframe, with emphasis on Ichimoku Cloud indicators',
    backstory="""You are an experienced cryptocurrency market analyst specializing in short-term Bitcoin analysis.
    Your expertise lies in technical analysis and identifying market trends in 30-minute charts.
    You are particularly skilled in interpreting Ichimoku Cloud indicators to predict market movements.
   You are known for your quick analysis and ability to spot rapid market changes.""",
    verbose=True,
    allow_delegation=False,
)

hourly_analyst = Agent(
    role=f'Hourly {symbol} Market Analyst',
    goal='Analyze Bitcoin market trends and patterns in 1-hour timeframe, with emphasis on Ichimoku Cloud indicators',
    backstory="""You are an experienced cryptocurrency market analyst specializing in short-term Bitcoin analysis.
    Your expertise lies in technical analysis and identifying market trends in hourly charts.
    You are particularly skilled in interpreting Ichimoku Cloud indicators to predict market movements.
    You are known for your balanced and objective analysis, considering both bullish and bearish scenarios.""",
    verbose=True,
    allow_delegation=False,
)

daily_analyst = Agent(
    role=f'6-Hourly {symbol} Market Analyst',
    goal='Analyze Bitcoin market trends and patterns in 6-hour timeframe',
    backstory="""You are an experienced cryptocurrency market analyst specializing in medium to long-term Bitcoin analysis.
    Your expertise lies in technical analysis and identifying market trends in daily charts.
    You are known for your cautious approach, always considering multiple market scenarios.""",
    verbose=True,
    allow_delegation=False,

)


strategist = Agent(
    role='Grid Trading Strategist',
    goal='Determine the most suitable grid trading strategy based on comprehensive market analysis and technical indicators',
    backstory="""You are a seasoned trading strategist with deep knowledge of various grid trading techniques, 
    particularly specializing in Bitcoin markets. Your expertise includes analyzing multiple timeframes, 
    interpreting Ichimoku Cloud signals, and distinguishing between trend reversals and pullbacks. 
    You excel at matching market conditions with appropriate trading strategies, considering factors 
    such as weighted timeframe analysis, consistency checks, and overall market trends. 
    You are known for your adaptive approach, often recommending a mix of strategies or regular grid 
    trading in uncertain markets, while confidently choosing directional strategies when clear trends emerge. 
    Your decisions are always backed by thorough analysis and clear explanations.""",
    verbose=True,
    allow_delegation=False,
)

price_predictor = Agent(
    role=f'{symbol} Multi-Timeframe Price Predictor',
    goal='Provide accurate price predictions for various timeframes while balancing short-term opportunities and longer-term trends',
    backstory="""You are an expert cryptocurrency analyst specializing in multi-timeframe price prediction. Your expertise spans from short-term,medium-term and long-term analyses, allowing you to provide comprehensive price forecasts.

    Your key strengths include:
    1. Synthesizing data from multiple timeframes (15-min, 30-min, 1-hour, 6-hour) to form cohesive price predictions.
    2. Utilizing a wide array of technical indicators, with a focus on leading indicators that signal potential future price movements.
    3. Identifying trend continuations, reversals, and breakout patterns across different timeframes.
    4. Incorporating volume analysis and market sentiment to enhance prediction accuracy.
    5. Providing specific price targets and confidence levels for different prediction horizons.
    6. Balancing short-term trading opportunities with awareness of longer-term market trends.
    7. Adapting predictions based on upcoming events or potential market catalysts.
    8. Maintaining a forward-looking perspective, always focusing on where the price is likely to go rather than where it has been.

    Your approach combines rigorous technical analysis with an understanding of market psychology and external factors. You're not afraid to make bold predictions when your analysis supports it, but you also clearly communicate levels of uncertainty.

    Your ultimate goal is to provide traders and investors with actionable insights that can inform their decision-making across various trading and investment horizons.""",
    verbose=True,
    allow_delegation=False,
)



korean_summarizer = Agent(
    role='Korean Market Summarizer',
    goal='Summarize Bitcoin market analysis and predictions in Korean',
    backstory="""You are a skilled translator and summarizer, specializing in cryptocurrency market analysis. 
    Your expertise lies in conveying complex market information in clear, concise Korean language. 
    You are known for your ability to make technical analysis accessible to Korean-speaking audiences.""",
    verbose=True,
    allow_delegation=False,

)


def extract_prediction(selected_strategy, result_string):
    # Determine direction based on selected_strategy
    if 'shortgrid' in selected_strategy.lower():
        direction = 'Down'
    elif 'longgrid' in selected_strategy.lower():
        direction = 'Up'
    elif 'regulargrid' in selected_strategy.lower():
        direction = 'Normal'
    else:
        direction = None

    # Extract percentage from result_string
    percentage_match = re.search(r'(\d+(?:\.\d+)?)%', result_string, re.IGNORECASE)

    if percentage_match:
        percentage = percentage_match.group(1)
    else:
        percentage = "0"

    return direction, percentage


def extract_strategy(text):
    strategies = ["RegularGrid", "ShortGrid", "LongGrid"]
    for strategy in strategies:
        if strategy in text:
            return strategy
    return None



def get_current_bitcoin_price(vt_symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={vt_symbol}"
        response = requests.get(url)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        print(f"Error fetching current Bitcoin price: {e}")
        return None


def get_strategy_config(strategy_name='240824'):
    try:
        strategy_config = StrategyConfig.objects.get(name=strategy_name)
        first_config = strategy_config.config.get('ETH', {})
        vt_symbol = first_config.get('vt_symbol', '')
        symbol = vt_symbol.split('.')[0] if vt_symbol else ''
        grid_strategy = first_config.get('setting', {}).get('grid_strategy')

        if not symbol or not grid_strategy:
            raise ValueError("Invalid configuration: missing symbol or grid_strategy")

        return {
            'vt_symbol': symbol,
            'grid_strategy': grid_strategy
        }
    except ObjectDoesNotExist:
        print(f"Strategy configuration not found for: {strategy_name}")
    except ValueError as e:
        print(f"Invalid configuration for {strategy_name}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in get_strategy_config: {str(e)}")



def get_bitcoin_data_from_api(symbol, max_retries=5, retry_delay=1):
    import time
    base_url = "https://gridtrade.one/api/v1/binanceData/llm-bitcoin-data/"
    params = {'symbol': symbol}

    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"최대 재시도 횟수 초과. API 호출 실패: {base_url}")
                raise

    return None  # 이 줄은 실행되지 않지만, 함수의 모든 경로에서 반환값이 있음을 보장합니다.


# TODO 이전의 RESULT STRING 값들 가져와서 추론 하기
# 기존 get_current_bitcoin_price 함수 내용...

def perform_eth_analysis():
    config = get_strategy_config()
    if not config:
        print("Strategy configuration is invalid.")
        return None

    vt_symbol = config['vt_symbol']
    grid_strategy = config['grid_strategy']

    print(f"Current grid_strategy: {grid_strategy}")

    # bitcoin_data = get_bitcoin_data(vt_symbol)
    bitcoin_data = get_bitcoin_data_from_api(vt_symbol)
    if not bitcoin_data:
        print("Failed to fetch bitcoin data.")
        return None

    task_15min = Task(
        description=f"""Analyze the Bitcoin market using the latest 12 hours of 15-minute data, The data is sorted from oldest to most recent, and each data point has the following structure:
        {bitcoin_data['15min'][-48:]}
        
        **Critical: Distinguish between a genuine trend reversal and a temporary pullback or retracement within the existing trend.**

        **Important:**
        - The **last row** of the data (`{bitcoin_data['15min'][-48:][-1]}`) is the **most recent data**.
        - When starting the analysis, begin with the last data point and proceed to analyze previous data points.

        Focus on:
        1. **Price Trends & Formations**: Identify upward or downward trends and any emerging trend patterns.
        2. **Volume Patterns**: Detect volume changes and their correlation with price movements.
        3. **Ichimoku Cloud Indicators**:
            - **Tenkan-sen & Kijun-sen Crossovers**: Compare 'Tenkan_sen' ({bitcoin_data['15min'][-48:][-1]['Tenkan_sen']}) and 'Kijun_sen' ({bitcoin_data['15min'][-48:][-1]['Kijun_sen']}) values to identify bullish (Tenkan > Kijun) or bearish (Tenkan < Kijun) signals.
            - **Senkou Span A vs B**: Compare Senkou Span A ({bitcoin_data['15min'][-48:][-1]['Senkou_Span_A']}) and Senkou Span B ({bitcoin_data['15min'][-48:][-1]['Senkou_Span_B']}). Check if they have recently crossed over, indicating a potential trend change.
            - **Price vs. Senkou Span A & B**: Compare 'close' price ({bitcoin_data['15min'][-48:][-1]['close']}) with 'Senkou_Span_A' ({bitcoin_data['15min'][-48:][-1]['Senkou_Span_A']}) and 'Senkou_Span_B' ({bitcoin_data['15min'][-48:][-1]['Senkou_Span_B']}) to determine if the price is above or below the cloud.
            - **Chikou Span Position**: If available, check if 'Chikou_Span' ({bitcoin_data['15min'][-48:][-1]['Chikou_Span']}) is above or below the current 'close' price ({bitcoin_data['15min'][-48:][-1]['close']}).
            - **Cloud (Kumo) Analysis**: 
                - Price vs Cloud: {bitcoin_data['15min'][-48:][-1]['close']} vs {max(bitcoin_data['15min'][-48:][-1]['Senkou_Span_A'], bitcoin_data['15min'][-48:][-1]['Senkou_Span_B'])} (top) / {min(bitcoin_data['15min'][-48:][-1]['Senkou_Span_A'], bitcoin_data['15min'][-48:][-1]['Senkou_Span_B'])} (bottom)
                - Analyze price-cloud interaction: Support at top or resistance at bottom?
                - Identify instances of price piercing cloud without closing outside (strong support/resistance).
                - Look for potential breakouts/breakdowns through the cloud.
                - Cloud Thickness: Current {abs(bitcoin_data['15min'][-48:][-1]['Senkou_Span_A'] - bitcoin_data['15min'][-48:][-1]['Senkou_Span_B'])}, Previous {abs(bitcoin_data['15min'][-48:][-2]['Senkou_Span_A'] - bitcoin_data['15min'][-48:][-2]['Senkou_Span_B'])}
                - Interpret thickness: Thick (strong trend, volatile) vs Thin (weak trend, easier breakouts)
                - Observe thickness changes over time for potential trend shifts.
                - Cloud Type: {'Bullish (Yang)' if bitcoin_data['15min'][-48:][-1]['Senkou_Span_A'] > bitcoin_data['15min'][-48:][-1]['Senkou_Span_B'] else 'Bearish (Yin)'}
                - Analyze cloud type and thickness for overall trend strength and future movement.


        4. **RSI and Stochastic Oscillator Analysis for Reversal Signals**: 
           - **RSI Analysis**: 
             - Look for RSI({bitcoin_data['15min'][-48:][-1]['RSI']}) values below 30 (oversold) or above 70 (overbought).
             - Identify potential bullish divergence (price making lower lows while RSI makes higher lows) or bearish divergence (price making higher highs while RSI makes lower highs).
           - **Stochastic Oscillator Analysis**:
             - Use '%K'({bitcoin_data['15min'][-48:][-1]['%K']}) and '%D'({bitcoin_data['15min'][-48:][-1]['%D']}) fields. 
             - Look for oversold conditions (both %K and %D below 20) or overbought conditions (both above 80).
             - Identify bullish crossovers (%K crossing above %D) in oversold territory or bearish crossovers in overbought territory.
           - **Combined RSI and Stochastic Analysis**:
             - Strong reversal signal: When both RSI and Stochastic indicate oversold/overbought conditions simultaneously.
             - Confirmation: Look for RSI starting to move away from oversold/overbought levels while Stochastic shows a crossover in the same direction.
        5. **Volume Confirmation**:
           - Check if volume increases as price starts to reverse, which can confirm the reversal.
           - Look for volume spikes that coincide with potential reversal candles.
        6. **Immediate Reversal Signals**:
           - **Oversold/Overbought Quick Recovery**: Look for rapid recoveries from oversold conditions in RSI or Stochastic.
           - **Volume Spikes**: Identify sudden increases in volume after a price drop, which might signal a reversal.
           - **Short-term Moving Average Crossovers**: 
             - Current 5MA: {bitcoin_data['15min'][-48:][-1]['MA5']}
             - Current 20MA: {bitcoin_data['15min'][-48:][-1]['MA20']}
             - Identify all instances where the 5MA crosses above or below the 20MA within the 48 data points.
             - Analyze the frequency and significance of these crossovers.

        Conclude with:
        - **Market Sentiment**: Bullish, Bearish, or Neutral based on the indicators.
        - **Reversal Potential**: High, Medium, or Low based on the combination of RSI, Stochastic, volume, and price action signals.
        - **Short-term Outlook**: 2-6 hours.
        - **Long-term Outlook**: 4-8 hours.
         - **Reversal Potential**: High, Medium, or Low based on the combination of all signals.
        - **Potential Reversal Points**: Identify key price levels where a reversal might occur.
        - **Cloud Dynamics**: Summarize the current state of the cloud, including its thickness, direction, and implications for future price movement.

        Conclude with:

        Ensure clarity and focus on key indicators to accurately determine market trends and potential immediate reversals.""",
        expected_output="Comprehensive Bitcoin market analysis report for the 15-minute timeframe, emphasizing key technical indicators and potential immediate reversal signals.",
        agent=fifteen_min_analyst
    )

    task_30min = Task(
        description=f"""Analyze the Bitcoin market using the latest 24 hours of 30-minute data, The data is sorted from oldest to most recent, and each data point has the following structure:
        {bitcoin_data['30min'][-48:]}
        
        **Critical: Distinguish between a genuine trend reversal and a temporary pullback or retracement within the existing trend.**

        **Important:**
        - The **last row** of the data (`{bitcoin_data['30min'][-48:][-1]}`) is the **most recent data**.
        - When starting the analysis, begin with the last data point and proceed to analyze previous data points.

        Focus on:
        1. **Price Trends & Formations**: Identify upward or downward trends and any emerging trend patterns.
        2. **Volume Patterns**: Detect volume changes and their correlation with price movements.
        3. **Ichimoku Cloud Indicators**:
            - **Tenkan-sen & Kijun-sen Crossovers**: Compare 'Tenkan_sen' ({bitcoin_data['30min'][-48:][-1]['Tenkan_sen']}) and 'Kijun_sen' ({bitcoin_data['30min'][-48:][-1]['Kijun_sen']}) values to identify bullish (Tenkan > Kijun) or bearish (Tenkan < Kijun) signals.
            - **Senkou Span A vs B**: Compare Senkou Span A ({bitcoin_data['30min'][-48:][-1]['Senkou_Span_A']}) and Senkou Span B ({bitcoin_data['30min'][-48:][-1]['Senkou_Span_B']}). Check if they have recently crossed over, indicating a potential trend change.
            - **Price vs. Senkou Span A & B**: Compare 'close' price ({bitcoin_data['30min'][-48:][-1]['close']}) with 'Senkou_Span_A' ({bitcoin_data['30min'][-48:][-1]['Senkou_Span_A']}) and 'Senkou_Span_B' ({bitcoin_data['30min'][-48:][-1]['Senkou_Span_B']}) to determine if the price is above or below the cloud.
            - **Chikou Span Position**: If available, check if 'Chikou_Span' ({bitcoin_data['30min'][-48:][-1]['Chikou_Span']}) is above or below the current 'close' price ({bitcoin_data['30min'][-48:][-1]['close']}).
            - **Cloud (Kumo) Analysis**: 
                - Price vs Cloud: {bitcoin_data['30min'][-48:][-1]['close']} vs {max(bitcoin_data['30min'][-48:][-1]['Senkou_Span_A'], bitcoin_data['30min'][-48:][-1]['Senkou_Span_B'])} (top) / {min(bitcoin_data['30min'][-48:][-1]['Senkou_Span_A'], bitcoin_data['30min'][-48:][-1]['Senkou_Span_B'])} (bottom)
                - Analyze price-cloud interaction: Support at top or resistance at bottom?
                - Identify instances of price piercing cloud without closing outside (strong support/resistance).
                - Look for potential breakouts/breakdowns through the cloud.
                - Cloud Thickness: Current {abs(bitcoin_data['30min'][-48:][-1]['Senkou_Span_A'] - bitcoin_data['30min'][-48:][-1]['Senkou_Span_B'])}, Previous {abs(bitcoin_data['30min'][-48:][-2]['Senkou_Span_A'] - bitcoin_data['30min'][-48:][-2]['Senkou_Span_B'])}
                - Interpret thickness: Thick (strong trend, volatile) vs Thin (weak trend, easier breakouts)
                - Observe thickness changes over time for potential trend shifts.
                - Cloud Type: {'Bullish (Yang)' if bitcoin_data['30min'][-48:][-1]['Senkou_Span_A'] > bitcoin_data['30min'][-48:][-1]['Senkou_Span_B'] else 'Bearish (Yin)'}
                - Analyze cloud type and thickness for overall trend strength and future movement.

        4. **RSI and Stochastic Oscillator Analysis for Reversal Signals**: 
           - **RSI Analysis**: 
             - Look for RSI({bitcoin_data['30min'][-48:][-1]['RSI']}) values below 30 (oversold) or above 70 (overbought).
             - Identify potential bullish divergence (price making lower lows while RSI makes higher lows) or bearish divergence (price making higher highs while RSI makes lower highs).
           - **Stochastic Oscillator Analysis**:
             - Use '%K'({bitcoin_data['30min'][-48:][-1]['%K']}) and '%D'({bitcoin_data['30min'][-48:][-1]['%D']}) fields. 
             - Look for oversold conditions (both %K and %D below 20) or overbought conditions (both above 80).
             - Identify bullish crossovers (%K crossing above %D) in oversold territory or bearish crossovers in overbought territory.
           - **Combined RSI and Stochastic Analysis**:
             - Strong reversal signal: When both RSI and Stochastic indicate oversold/overbought conditions simultaneously.
             - Confirmation: Look for RSI starting to move away from oversold/overbought levels while Stochastic shows a crossover in the same direction.
    
        5. **Volume Confirmation**:
           - Check if volume increases as price starts to reverse, which can confirm the reversal.
           - Look for volume spikes that coincide with potential reversal candles.
    
        6. **Price Action Patterns**:
           - Identify reversal candlestick patterns such as hammer, inverted hammer, engulfing patterns, or doji in oversold/overbought conditions.
           - Look for double bottoms or double tops that might indicate a potential reversal.
    
        Conclude with:
        - **Market Sentiment**: Bullish, Bearish, or Neutral based on the indicators.
        - **Reversal Potential**: High, Medium, or Low based on the combination of RSI, Stochastic, volume, and price action signals.
        - **Short-term Outlook**: 4-8 hours.
        - **Long-term Outlook**: 6-12 hours.
        - **Potential Reversal Points**: Identify key price levels where a reversal might occur.

        Conclude with:

        Ensure clarity and focus on key indicators to accurately determine market trends.""",
        expected_output="Concise and accurate Bitcoin market analysis report for the 30-minute timeframe, emphasizing key Ichimoku Cloud signals and other technical indicators.",
        agent=thirty_min_analyst
    )

    task1 = Task(
        description=f"""Use the latest 72 hours of hourly data to analyze the Bitcoin market. The data is sorted from oldest to most recent, and each data point has the following structure:
        {bitcoin_data['hourly'][-72:]}

        **Critical: Distinguish between a genuine trend reversal and a temporary pullback or retracement within the existing trend.**

        **Important:**
        - The **last row** of the data (`{bitcoin_data['hourly'][-72:][-1]}`) is the **most recent data**.
        - When starting the analysis, begin with the last data point and proceed to analyze previous data points.

        Focus on:
        1. **Price Trends**: Identify upward or downward trends starting from the latest price.
        2. **Volume Changes**: Compare recent volumes with earlier ones to gauge market interest.
        3. **Ichimoku Cloud Indicators**:
            - **Kumo (Cloud) Position**: Is the price ({bitcoin_data['hourly'][-1]['close']}) above or below the cloud (Senkou Span A: {bitcoin_data['hourly'][-1]['Senkou_Span_A']}, Senkou Span B: {bitcoin_data['hourly'][-1]['Senkou_Span_B']})?
            - **Senkou Span A vs B**: Compare Senkou Span A ({bitcoin_data['hourly'][-1]['Senkou_Span_A']}) and Senkou Span B ({bitcoin_data['hourly'][-1]['Senkou_Span_B']}). Check if they have recently crossed over, indicating a potential trend change.
            - **Tenkan-sen vs. Kijun-sen**: Look for crossovers (Bullish or Bearish signals). Tenkan-sen: {bitcoin_data['hourly'][-1]['Tenkan_sen']}, Kijun-sen: {bitcoin_data['hourly'][-1]['Kijun_sen']}.
            - **Chikou Span**: Position relative to the current price. Chikou Span: {bitcoin_data['hourly'][-1]['Chikou_Span']}, Current price: {bitcoin_data['hourly'][-1]['close']}.
            - **Cloud (Kumo) Analysis**:
                - Price vs Cloud: {bitcoin_data['hourly'][-1]['close']} vs {max(bitcoin_data['hourly'][-1]['Senkou_Span_A'], bitcoin_data['hourly'][-1]['Senkou_Span_B'])} (top) / {min(bitcoin_data['hourly'][-1]['Senkou_Span_A'], bitcoin_data['hourly'][-1]['Senkou_Span_B'])} (bottom)
                - Price-cloud interaction: Support at top or resistance at bottom?
                - Instances of price piercing cloud without closing outside (strong support/resistance).
                - Potential breakouts/breakdowns through the cloud.
                - Cloud Thickness: Current {abs(bitcoin_data['hourly'][-1]['Senkou_Span_A'] - bitcoin_data['hourly'][-1]['Senkou_Span_B'])}, Previous {abs(bitcoin_data['hourly'][-2]['Senkou_Span_A'] - bitcoin_data['hourly'][-2]['Senkou_Span_B'])}
                - Interpret thickness: Thick (strong trend, volatile) vs Thin (weak trend, easier breakouts)
                - Cloud Type: {'Bullish (Yang)' if bitcoin_data['hourly'][-1]['Senkou_Span_A'] > bitcoin_data['hourly'][-1]['Senkou_Span_B'] else 'Bearish (Yin)'}
                - Analyze cloud type and thickness for overall trend strength and future movement.


        4. **RSI and Stochastic Oscillator Analysis for Reversal Signals**:
           - **RSI Analysis**:
             - Look for RSI({bitcoin_data['hourly'][-1]['RSI']}) values below 30 (oversold) or above 70 (overbought).
             - Identify potential bullish divergence (price making lower lows while RSI makes higher lows) or bearish divergence (price making higher highs while RSI makes lower highs).
           - **Stochastic Oscillator Analysis**:
             - Use '%K'({bitcoin_data['hourly'][-1]['%K']}) and '%D'({bitcoin_data['hourly'][-1]['%D']}) fields.
             - Look for oversold conditions (both %K and %D below 20) or overbought conditions (both above 80).
             - Identify bullish crossovers (%K crossing above %D) in oversold territory or bearish crossovers in overbought territory.
           - **Combined RSI and Stochastic Analysis**:
             - Strong reversal signal: When both RSI and Stochastic indicate oversold/overbought conditions simultaneously.
             - Confirmation: Look for RSI starting to move away from oversold/overbought levels while Stochastic shows a crossover in the same direction.
        5. **Technical Patterns**: Detect formations like Head and Shoulders, Double Tops/Bottoms, etc.

        Conclude with:
        - **Market Sentiment**: Bullish, Bearish, or Neutral based on the above indicators.
        - **Short-term Outlook**: 12-24 hours.
        - **Long-term Outlook**: 1-3 days.

        Ensure clarity and focus on key indicators to accurately determine market trends.""",
        expected_output="Concise and accurate Bitcoin market analysis report for the 1-hour timeframe, emphasizing key Ichimoku Cloud signals and other technical indicators.",
        agent=hourly_analyst
    )

    task2 = Task(
        description=f"""Analyze the Bitcoin market using the latest 30 days of 6-hourly data, with a strong emphasis on identifying and characterizing the current trend. The data is sorted from oldest to most recent, and each data point has the following structure:
        {bitcoin_data['daily'][-180:]}

        **Critical: Distinguish between a genuine trend reversal and a temporary pullback or retracement within the existing trend.**

        **Important:**
        - The **last row** of the data (`{bitcoin_data['daily'][-180:][-1]}`) is the **most recent data**.
        - When starting the analysis, begin with the last data point and proceed to analyze previous data points.
        - Determine the primary trend (bullish, bearish, or neutral) based on the overall price movement over the 30-day period.
        - Identify any potential trend reversals or continuations in the most recent data points.


        Focus on:
        1. **Price Trends & Key Levels**: Identify upward or downward trends and significant support/resistance levels.
        2. **Volume Patterns**: Detect significant volume spikes and their correlation with price movements.
        3. **Ichimoku Cloud Indicators**:
            - Price vs Cloud: {bitcoin_data['daily'][-1]['close']} vs {max(bitcoin_data['daily'][-1]['Senkou_Span_A'], bitcoin_data['daily'][-1]['Senkou_Span_B'])} (top) / {min(bitcoin_data['daily'][-1]['Senkou_Span_A'], bitcoin_data['daily'][-1]['Senkou_Span_B'])} (bottom)
            - Senkou Span A ({bitcoin_data['daily'][-1]['Senkou_Span_A']}) vs B ({bitcoin_data['daily'][-1]['Senkou_Span_B']})
              Check for recent crossovers indicating potential trend changes.
            - Tenkan-sen ({bitcoin_data['daily'][-1]['Tenkan_sen']}) vs Kijun-sen ({bitcoin_data['daily'][-1]['Kijun_sen']})
              Identify bullish (Tenkan > Kijun) or bearish (Tenkan < Kijun) signals.
            - Chikou Span ({bitcoin_data['daily'][-1]['Chikou_Span']}) vs Price ({bitcoin_data['daily'][-1]['close']})
            - **Cloud (Kumo) Analysis**:
                - Price-cloud interaction: Support at top or resistance at bottom?
                - Instances of price piercing cloud without closing outside (strong support/resistance).
                - Potential breakouts/breakdowns through the cloud.
                - Cloud Thickness: Current {abs(bitcoin_data['daily'][-1]['Senkou_Span_A'] - bitcoin_data['daily'][-1]['Senkou_Span_B'])}, Previous {abs(bitcoin_data['daily'][-2]['Senkou_Span_A'] - bitcoin_data['daily'][-2]['Senkou_Span_B'])}
                - Interpret thickness: Thick (strong trend, volatile) vs Thin (weak trend, easier breakouts)
                - Thickness changes over time for potential trend shifts.
                - Cloud Type: {'Bullish (Yang)' if bitcoin_data['hourly'][-1]['Senkou_Span_A'] > bitcoin_data['hourly'][-1]['Senkou_Span_B'] else 'Bearish (Yin)'}
                - Analyze cloud type and thickness for overall trend strength and future movement.


        4. **RSI and Stochastic Oscillator Analysis for Reversal Signals**:
           - **RSI Analysis**:
             - Look for RSI({bitcoin_data['daily'][-1]['RSI']}) values below 30 (oversold) or above 70 (overbought).
             - Identify potential bullish divergence (price making lower lows while RSI makes higher lows) or bearish divergence (price making higher highs while RSI makes lower highs).
           - **Stochastic Oscillator Analysis**:
             - Use '%K'({bitcoin_data['daily'][-1]['%K']}) and '%D'({bitcoin_data['daily'][-1]['%D']}) fields.
             - Look for oversold conditions (both %K and %D below 20) or overbought conditions (both above 80).
             - Identify bullish crossovers (%K crossing above %D) in oversold territory or bearish crossovers in overbought territory.
           - **Combined RSI and Stochastic Analysis**:
             - Strong reversal signal: When both RSI and Stochastic indicate oversold/overbought conditions simultaneously.
             - Confirmation: Look for RSI starting to move away from oversold/overbought levels while Stochastic shows a crossover in the same direction.


        Conclude with:
        - **Market Sentiment**: Bullish, Bearish, or Neutral based on the indicators.
        - **Short-term Outlook**: 1-3 days.
        - **Long-term Outlook**: 3-7 days.

        Ensure clarity and focus on key indicators to accurately determine market trends.""",
        expected_output="Concise and accurate Bitcoin market analysis report for the 1-day timeframe, emphasizing key Ichimoku Cloud signals and other technical indicators.",
        agent=daily_analyst
    )

    bitcoin_analysis = Crew(
        agents=[fifteen_min_analyst, thirty_min_analyst, hourly_analyst, daily_analyst],
        tasks=[task_15min, task_30min, task1, task2],
        verbose=True,
        process=Process.sequential
    )
    analysis_crew = bitcoin_analysis.kickoff()
    analysis_results = {}
    analysis_results['15min'] = analysis_crew.tasks_output[0]
    analysis_results['30min'] = analysis_crew.tasks_output[1]
    analysis_results['1hour'] = analysis_crew.tasks_output[2]
    analysis_results['daily'] = analysis_crew.tasks_output[3]
    task3 = Task(
        description=f"""Based on the analyses from the **15-minute, 30-minute , 1-hour, 6-hour timeframes**, forecast Bitcoin price movements for the next:

        **15-minute Analysis**:
        {analysis_results['15min']}

        **30-minute Analysis**:
        {analysis_results['30min']}

        **1-hour Analysis**:
        {analysis_results['1hour']}

        **6-hour Analysis**:
        {analysis_results['daily']}

        1. **2-6 hours (Very Short-term)**
        2. **6-24 hours (Short-term)**
        3. **1-3 days (Medium-term)**
        4. **3-7 days (Long-term)**

        **CRITICAL FOCUS: PREDICTIVE ANALYSIS**
        Your primary goal is to provide accurate, forward-looking price predictions. Emphasis should be on identifying potential future price movements rather than summarizing past performance. Use the provided analyses as a foundation, but focus on extrapolating future trends and potential price targets.

        For each timeframe, provide:

        1. **Directional Prediction**: Determine the most likely direction (Up/Down/Neutral) using a forward-looking approach that considers potential market catalysts and trend continuations or reversals.
        2. **Confidence Level**: Assign a confidence percentage based on the strength of predictive indicators and potential for trend continuation or reversal.
        3. **Price Targets**: Provide specific price levels that Bitcoin might reach within the given timeframe, including potential breakout or breakdown points.
        4. **Key Technical Levels**: Highlight crucial future support and resistance levels that may influence price movement.

       **Guidelines:**  
        - **Balanced Analysis Approach**: Give equal weight to all timeframe analyses for a comprehensive view.
        - **Trend Alignment**: 
            - If trends across timeframes align, increase confidence level.
            - If they conflict, decrease confidence and explain the discrepancy.
        - **Trend Anticipation**: Look for early signs of trend reversals or continuations, such as chart patterns forming or key level tests approaching.
        - **Cross-Timeframe Confirmation**: Seek alignment of predictive signals across multiple timeframes for stronger forecasts.
        - **Technical Indicator Emphasis**: Pay special attention to Ichimoku Cloud positions, RSI levels, and volume patterns across all timeframes.
        - **Market Sentiment Consideration**: Factor in the overall market sentiment described in each timeframe analysis.
        - **Adaptive Prediction**: 
            - For shorter timeframes, consider immediate market conditions more heavily.
            - For longer timeframes, give more weight to overarching trends from the 6-hour analysis.     
        - **Neutral Stance**: If indicators are mixed or unclear, do not hesitate to predict a neutral direction.
        - **Conciseness**: Keep predictions clear and to the point.
        - **Regular Assessment**: Continuously evaluate the effectiveness of predictions and adjust the strategy as needed.

        **Output Format:**

        End your response with three lines indicating the predicted direction and confidence level for each timeframe:
             "2-6 hours: [Up/Down/Neutral] [Confidence]% | Target Range: $[Low] - $[High]"
             "6-24 hours: [Up/Down/Neutral] [Confidence]% | Target Range: $[Low] - $[High]"
             "1-3 days: [Up/Down/Neutral] [Confidence]% | Target Range: $[Low] - $[High]"
             "3-7 days: [Up/Down/Neutral] [Confidence]% | Target Range: $[Low] - $[High]"

        """,
        expected_output="Accurate, forward-looking Bitcoin price predictions with directional outcomes, confidence levels, and specific price targets for short-term, medium-term, and longer-term timeframes, emphasizing predictive analysis and potential future market behavior.",
        agent=price_predictor

    )

    # - ** Focus  on  Short - Term   Timeframes **: Use the  15 - minute, 30 - minute and 1 - hour     analyses as the    primary    basis   for all predictions.

    task4 = Task(
        description=f"""Determine the most suitable grid trading strategy (**RegularGrid**, **ShortGrid**, **LongGrid**) for Bitcoin based on the predictions from analysis for 2-6 hours, 6-24 hours, 1-3 days, 3-7 days and timeframes, as well as Ichimoku Cloud signals and technical indicators.

        **Guidelines:**
        1. Weighted Timeframe Analysis:
           Calculate a weighted average confidence for each direction (Up/Down) using the following weights:
           - 2-6 hours: Weight 1
           - 6-24 hours: Weight 2
           - 1-3 days: Weight 3
           - 3-7 days: Weight 4

           Calculation method:
           - For 'Up' direction: 
             Sum (confidence * weight for 'Up' predictions + half of confidence * weight for 'Neutral' predictions) / 10
           - For 'Down' direction: 
             Sum (confidence * weight for 'Down' predictions + half of confidence * weight for 'Neutral' predictions) / 10

        2. Consistency Check:
           - If all timeframes show the same direction (all 'Up' or all 'Down'), increase the confidence in the selected strategy by 12%.
           - If the two longest timeframes (1-3 days and 3-7 days) show the same direction, increase the confidence in the selected strategy by 7%.

        3. Ichimoku Cloud Analysis:
           - Analyze the position and thickness of the Ichimoku Cloud across all timeframes.
           - If the price is above the cloud or the cloud is providing support:
               - Increase the probability of LongGrid by 5% (e.g., from 65% to 70%)
           - If the price is below the cloud or the cloud is acting as resistance:
               - Increase the probability of ShortGrid by 5% (e.g., from 65% to 70%)
           - Consider cloud thickness:
               - Thick cloud: Suggests stronger support/resistance and potentially more stable trends
               - Thin cloud: Indicates potential for easier breakouts/breakdowns and trend changes

        4. Strategy Selection Criteria:
           - LongGrid:
               - Base case: Select if the weighted average confidence for 'Up' direction is 70% or higher.
               - Cloud-adjusted case: If the price is above the cloud or the cloud is providing support, select LongGrid with a weighted average confidence of 60% or higher for 'Up' direction.
           - ShortGrid:
               - Base case: Select if the weighted average confidence for 'Down' direction is 70% or higher.
               - Cloud-adjusted case: If the price is below the cloud or the cloud is acting as resistance, select ShortGrid with a weighted average confidence of 60% or higher for 'Down' direction.
           - RegularGrid:
               - Select if predictions are mixed or the weighted average confidence is below the specified thresholds for both LongGrid and ShortGrid, even after cloud adjustments.

        5. Distinguish between Trend Reversal and Pullback:
           - Carefully assess whether recent price movements indicate a true reversal or merely a pullback within the existing trend.
           - Consider the strength and duration of the current trend when making this assessment.

        6. Overall Trend Consideration:
           - Factor in the general market sentiment and trend from the 6-hour analysis when making the final decision.
           - Pay special attention to the cloud dynamics in the 6-hour timeframe for longer-term trend indication.

        7. Maximize Profit Potential:
           - Choose the strategy that best aligns with the overall market direction and has the highest probability of profit.
           - Consider the potential for trend continuation vs. reversal based on cloud analysis and other technical indicators.

        **Output Format:**

        Provide a brief explanation of your decision, including:
        1. The weighted average confidence for each direction (Up/Down)
        2. Key factors from the Ichimoku Cloud analysis
        3. Whether recent movements are likely a trend reversal or a pullback
        4. The final selected strategy and its confidence level after all adjustments

        Then, at the end of your response, provide a single word: 'RegularGrid', 'ShortGrid', or 'LongGrid'.
        """,
        expected_output="""A concise explanation of the grid trading strategy recommendation, including weighted average confidences, key Ichimoku Cloud insights, trend reversal vs pullback assessment, and the final strategy selection with confidence level, followed by the chosen strategy: 'RegularGrid', 'ShortGrid', or 'LongGrid'.""",
        agent=strategist
    )
    # Crew 인스턴스화
    crew = Crew(
        agents=[price_predictor, strategist],
        tasks=[task3, task4],
        verbose=True,
        process=Process.sequential
    )

    results = crew.kickoff()
    print("CrewOutput type:", type(results))
    result_string = str(results)
    ##type: <class 'crewai.crews.crew_output.CrewOutput'>

    print("-----------------------------------------------------")
    print(results)  # results.raw)
    print("-----------------------------------------------------")

    task_results = {
        '15min_analysis': analysis_results['15min'],
        '30min_analysis': analysis_results['30min'],
        'hourly_analysis': analysis_results['1hour'],
        'daily_analysis': analysis_results['daily'],
        'price_prediction': results.tasks_output[0],
        'strategy_recommendation': results.tasks_output[1]
    }

    selected_strategy = extract_strategy(result_string)
    price_prediction, confidence = extract_prediction(selected_strategy, result_string)
    # current_price = get_current_bitcoin_price(vt_symbol)
    current_price = 0
    korean_summary_task = Task(
        description=f"""Summarize the following Bitcoin market analysis in Korean:
        1. 15-minute Analysis: {task_results['15min_analysis']}
        2. 30-Minute Analysis: {task_results['30min_analysis']}
        3. Hourly Analysis: {task_results['hourly_analysis']}
        4. 6-Hourly Analysis: {task_results['daily_analysis']}
        5. Price Prediction: {task_results['price_prediction']}
        6. Strategy Recommendation: {task_results['strategy_recommendation']}

        Price Prediction: {price_prediction}
        Confidence: {confidence}%

        Provide a detailed summary in Korean, highlighting the key points from each analysis. Explain any technical terms if necessary.
        The 30-minute, hourly, and 6-hourly analyses, as well as the Price Prediction and Probability Assessments must be analyzed and presented separately in detail.

        **Additionally, include a summary of the Ichimoku Cloud analysis based on the data from the 30-minute, hourly, and 6-hourly analyses.**

        Translate the final conclusion and selected strategy as follows:

        ★ Final Conclusion: {result_string}
        ★ Selected Strategy: {selected_strategy}

       IMPORTANT: Structure your response clearly and elegantly using the following format:

        1. Use Markdown headers (##) for each main section:
           - 15분 분석
           - 30분 분석
           - 시간별 분석
           - 6시간 분석
           - 가격 예측
           - 확률 평가
           - 전략 추천
           - **일목균형표 분석**
           - 주요 지표
           - 최종 결론
           - 선택된 전략
        2. Use bullet points or numbered lists for key points within each section.
        3. Highlight important information using bold text or symbols.
        4. Present the 주요 지표 (Key Indicators) section as a list with clear labels.
        5. Use the ★ symbol before the Final Conclusion and Selected Strategy.
        6. Add a horizontal rule (---) after each section to clearly separate them.

        Ensure that your summary is easy to read at a glance, with clear separation between sections and emphasis on crucial information.""",
        expected_output="A well-structured, clear, and concise Korean summary of the Bitcoin market analysis and predictions, including a summary of Ichimoku Cloud analysis, with translated final conclusion and selected strategy, formatted for easy readability and clear section separation",
        agent=korean_summarizer
    )

    # 한글 요약을 위한 새로운 Crew 생성 및 실행
    korean_summary_crew = Crew(
        agents=[korean_summarizer],
        tasks=[korean_summary_task],
        verbose=True,
        process=Process.sequential
    )

    korean_summary_result = korean_summary_crew.kickoff()
    korean_summary = str(korean_summary_result)

    print("######################")
    print("분석 완료")
    print(f"선택된 그리드 전략: {selected_strategy}")
    print(f"가격 예측: {price_prediction}")
    print(f"신뢰도: {confidence}%")
    print("\n한글 요약:")
    print(korean_summary)

    return {
        'symbol': vt_symbol,
        'result_string': result_string,
        'price_prediction': price_prediction,
        'confidence': confidence,
        'selected_strategy': selected_strategy,
        'current_price': current_price,
        'korean_summary': korean_summary,
        'analysis_results_30m': analysis_results['30min'],
        'analysis_results_1hour': analysis_results['1hour'],
        'analysis_results_daily': analysis_results['daily']
    }
