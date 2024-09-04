
import os
from crewai import Agent, Task, Crew, Process
from binance.client import Client
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

# Binance 클라이언트 초기화
symbol = "BNBUSDT"

binance_api_key = settings.BINANCE_API_KEY
binance_api_secret = settings.BINANCE_API_SECRET
client = Client(binance_api_key, binance_api_secret)
cryptocompare_api_key = "400daae3cf09044e5d78b3fc744b107731547031372de5573431166b96d16db7"

def get_crypto_news():
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={cryptocompare_api_key}"
    response = requests.get(url)
    news_data = response.json()['Data']
    return news_data[:30]  # 최근 10개의 뉴스만 반환

def get_bitcoin_data(symbol):
    try:
        end_date = datetime.now()
        start_date_hourly = end_date - timedelta(days=21)  # 약 500시간
        start_date_daily = end_date - timedelta(days=500)  # 500일

        # 1시간 및 1일 간격의 데이터 가져오기
        hourly_candles = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, start_date_hourly.strftime("%d %b %Y %H:%M:%S"), end_date.strftime("%d %b %Y %H:%M:%S"))
        daily_candles = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start_date_daily.strftime("%d %b %Y %H:%M:%S"), end_date.strftime("%d %b %Y %H:%M:%S"))

        def process_candles(candles):
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                                'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

            for ma in [5, 10, 20, 24, 50, 100, 200]:
                df[f"MA{ma}"] = df["close"].rolling(window=ma).mean()
            period = 20
            multiplier = 2.0
            df["MA"] = df["close"].rolling(window=period).mean()
            df["STD"] = df["close"].rolling(window=period).std()
            df["Upper"] = df["MA"] + (df["STD"] * multiplier)
            df["Lower"] = df["MA"] - (df["STD"] * multiplier)
            StochasticRSI(df)
            RSIAnalyzer(df)
            return df.to_dict(orient='records')

        return {
            'hourly': process_candles(hourly_candles),
            'daily': process_candles(daily_candles)
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

news_analyst = Agent(
    role='Crypto News Trend Analyst',
    goal='Analyze recent cryptocurrency news to predict short-term and long-term Bitcoin price trends',
    backstory="""You are a seasoned crypto news analyst with an exceptional ability to predict market trends. 
    Your expertise lies in quickly digesting news information and translating it into actionable trend forecasts for Bitcoin. 
    You have a proven track record of accurately predicting both short-term (12-24 hours) and long-term (1-2 days) price movements based on news sentiment and market-moving events.""",
    verbose=True,
    allow_delegation=False
)


hourly_analyst = Agent(
    role=f'Hourly {symbol} Market Analyst',
    goal='Analyze Bitcoin market trends and patterns in 1-hour timeframe',
    backstory="""You are an experienced cryptocurrency market analyst specializing in short-term Bitcoin analysis.
    Your expertise lies in technical analysis and identifying market trends in hourly charts.
    You are known for your balanced and objective analysis, considering both bullish and bearish scenarios.""",
    verbose=True,
    allow_delegation=False,
)

daily_analyst = Agent(
    role=f'Daily {symbol} Market Analyst',
    goal='Analyze Bitcoin market trends and patterns in 1-day timeframe',
    backstory="""You are an experienced cryptocurrency market analyst specializing in medium to long-term Bitcoin analysis.
    Your expertise lies in technical analysis and identifying market trends in daily charts.
    You are known for your cautious approach, always considering multiple market scenarios.""",
    verbose=True,
    allow_delegation=False,
)

strategist = Agent(
    role='Grid Trading Strategist',
    goal='Determine the most suitable grid trading strategy based on market analysis',
    backstory="""You are a seasoned trading strategist with deep knowledge of various grid trading techniques.
    You excel at matching market conditions with appropriate trading strategies.
    You are known for your adaptive approach, often recommending a mix of strategies or regular grid trading in uncertain markets.""",
    verbose=True,
    allow_delegation=False,
)

price_predictor = Agent(
    role=f'{symbol} Balanced Intraday Futures Trader',
    goal='Maximize intraday profits in cryptocurrency futures while minimizing liquidation risks',
    backstory="""You are a seasoned intraday trader specializing in cryptocurrency futures, known for your ability to balance high returns with prudent risk management.
    Your expertise lies in identifying profitable short-term opportunities while maintaining a keen awareness of liquidation risks.
    You employ a combination of technical analysis, market sentiment tracking, and risk assessment tools to make informed trading decisions.
    Your approach is characterized by strategic use of leverage, active position management, and swift response to market changes.
    While profit maximization remains a priority, you never compromise on protecting your capital from liquidation events.""",
    verbose=True,
    allow_delegation=False,
)


# price_predictor = Agent(
#     role=f'{symbol} Price Predictor',
#     goal='Predict the future price movement of Bitcoin and provide a confidence level',
#     backstory="""You are an expert in price prediction for cryptocurrencies, especially Bitcoin.
#     You use a combination of technical analysis, market sentiment, and historical patterns to make educated guesses about future price movements.
#     You are known for your conservative estimates and rarely give extremely high confidence levels.""",
#     verbose=True,
#     allow_delegation=False,
# )

korean_summarizer = Agent(
    role='Korean Market Summarizer',
    goal='Summarize Bitcoin market analysis and predictions in Korean',
    backstory="""You are a skilled translator and summarizer, specializing in cryptocurrency market analysis. 
    Your expertise lies in conveying complex market information in clear, concise Korean language. 
    You are known for your ability to make technical analysis accessible to Korean-speaking audiences.""",
    verbose=True,
    allow_delegation=False,
)


def extract_prediction(text):
    # Look for the specific format: "Up X%" or "Down X%"
    match = re.search(r'(Up|Down)\s+(\d+(?:\.\d+)?)%', text, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)

    percentage_match = re.search(r'(\d+(?:\.\d+)?)%', text, re.IGNORECASE)

    if percentage_match:
        return "Down", percentage_match.group(1)

    return None, None


def extract_strategy(text):
    strategies = ["RegularGrid", "ShortGrid", "LongGrid"]
    for strategy in strategies:
        if strategy in text:
            return strategy
    return None


def get_current_bitcoin_price(vt_symbol):
    try:
        ticker = client.get_symbol_ticker(symbol=vt_symbol)
        return float(ticker['price'])
    except BinanceAPIException as e:
        print(f"Error fetching current Bitcoin price: {e}")
        return None


def get_strategy_config(strategy_name='240824'):
    try:
        strategy_config = StrategyConfig.objects.get(name=strategy_name)
        first_config = strategy_config.config.get('INIT', {})
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


def get_bitcoin_data_from_api(symbol):
    # url = f"{settings.MAIN_SERVER_URL}/api/v1/binanceData/llm-bitcoin-data/{symbol}/"
    url = f"https://gridtrade.one/api/v1/binanceData/llm-bitcoin-data/{symbol}/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"API 호출 실패: {response.status_code}")
        return None



# 기존 get_current_bitcoin_price 함수 내용...

def perform_analysis():
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

    task_news = Task(
        description=f"""Analyze the following recent cryptocurrency news items and determine their potential impact on Bitcoin's price trend:
        {[{
            'title': news['title'],
            'content': news['body'],
        } for news in get_crypto_news()]}

        1. Quickly review all news items, focusing on their potential impact on Bitcoin's price.
        2. Identify key themes or events that could significantly influence Bitcoin's price.
        3. Based on the overall sentiment of the news, determine:
           a) The likely short-term trend (next 24-48 hours): Bullish, Bearish, or Neutral
           b) The potential long-term trend (next 1-2 weeks): Bullish, Bearish, or Neutral

        Provide a concise summary (3-5 sentences) explaining your trend predictions, referencing the most impactful news items.

        End your analysis with two lines:
        "NEWS_Short-term trend: [Bullish/Bearish/Neutral]"
        "NEWS_Long-term trend: [Bullish/Bearish/Neutral]"
        """,
        expected_output="Concise analysis of Bitcoin's likely price trends based on recent news, with clear short-term and long-term trend predictions.",
        agent=news_analyst
    )


    # 태스크 생성
    task1 = Task(
        description=f"""Conduct a comprehensive analysis of the Bitcoin market using the most recent 72 hours of hourly data:
        {bitcoin_data['hourly'][-72:]}

        IMPORTANT: Start your analysis from the most recent data point (the last entry in the provided dataset) and work backwards.

        Focus on the most recent 72 hours, examining:
        1. Recent price trends (start from the latest price)
        2. Volume changes (compare recent volumes to earlier ones)
        3. RSI (Relative Strength Index) - Focus on the latest readings. Pay close attention to overbought (RSI > 70) and oversold (RSI < 30) conditions. These levels often indicate potential price reversals or consolidations.
        4. Bollinger Bands (using the 'Upper', 'Lower', and 'MA' columns) - analyze recent price positions relative to the bands
        5. Stochastic oscillator - Strongly emphasize that trend reversals occur when %K and %D lines cross each other. Pay special attention to these crossover points as they may indicate potential trend changes.
        6. Technical patterns - look for and analyze recent formations of patterns such as:
           - Triangle patterns (ascending, descending, symmetrical)
           - Head and shoulders
           - Double tops/bottoms
           - Flags and pennants
        Identify significant support and resistance levels, and overall market sentiment in the 1-hour timeframe based on the most recent data.
        Consider both bullish and bearish scenarios in your analysis, with emphasis on the current market conditions.
        Conclude with an overall market outlook for the short-term (6-12 hours) and long-term (12-24 hours) based on this analysis, with particular attention to the most recent market developments.""",
        expected_output="Detailed Bitcoin market analysis report for 1-hour timeframe, focusing on the most recent market conditions",
        agent=hourly_analyst
    )

    task2 = Task(
        description=f"""Conduct a comprehensive analysis of the Bitcoin market using the most recent 90 days of daily data:
        {bitcoin_data['daily'][-90:]}

        IMPORTANT: Begin your analysis with the most recent data point (the last entry in the dataset) and work backwards through time.

        Focus on the most recent 90 days, examining:
        1. Price trends and key price levels (start from the latest price)
        2. Volume patterns and significant volume spikes (focus on recent volume activity)
        3. RSI (Relative Strength Index) - Focus on the latest readings. Pay close attention to overbought (RSI > 70) and oversold (RSI < 30) conditions. These levels often indicate potential price reversals or consolidations.
        4. Stochastic oscillator - Strongly emphasize that trend reversals occur when %K and %D lines cross each other. Pay special attention to these crossover points as they may indicate potential trend changes.
        5. Support and resistance levels - identify key levels based on recent price action and MA
        6. Technical patterns - look for and analyze recent formations of patterns such as:
           - Triangle patterns (ascending, descending, symmetrical)
           - Head and shoulders
           - Double tops/bottoms
           - Flags and pennants
        7. Overall market sentiment based on the above indicators and patterns, with emphasis on the current market state

        Provide a balanced analysis considering both bullish and bearish scenarios, focusing on the present market conditions. 
        Highlight any significant recent divergences between price action and indicators.
        Conclude with an overall market outlook for the short-term (12-24 hours) and long-term (1-3 days) based on this analysis, with particular attention to the most recent market developments.""",
        expected_output="Detailed Bitcoin market analysis report for the most recent 90 days (1-day timeframe), including technical patterns and market outlook, with emphasis on current market conditions",
        agent=daily_analyst
    )

    task3 = Task(
        description="""Predict future Bitcoin price scenarios for the next 6-24 hours (short-term) and 1-3 days (long-term):

        1. Describe one bullish and one bearish scenario for each timeframe
        2. Include specific price targets or ranges for each scenario
        3. Identify immediate potential triggers or catalysts for each scenario
        4. Assign probabilities to each scenario (ensure they sum to 100% per timeframe)
        5. Highlight key technical levels to watch in the very near term

        Focus on forecasting immediate future developments, emphasizing short-term trading perspectives. Consider:
        - Rapid market sentiment shifts
        - RSI overbought, oversold and stochastic oscillator crossover points and Technical patterns
        - Immediate changes in trading patterns and volume
        
        Based on your analysis, provide a single most likely direction for the next 6-24 hours.
        End your response with either 'Up' or 'Down' followed by the confidence percentage, e.g., 'Up 80%', 'Down 85%', 'Up 70%' or 'Down 65%'.""",
        expected_output="Concise short-term future scenario analysis for Bitcoin with a single directional prediction and confidence level",
        agent=price_predictor
    )

    task4 = Task(
        description="""Based on the market analyses provided for both 1-hour and 1-day timeframes, and considering the price prediction,
        determine the most suitable grid trading strategy among regular grid, short grid, and long grid. 
        Provide a clear rationale for your choice, considering both short-term and long-term market conditions.
        Use the following strict guidelines:
        - If the price prediction is 'Up' with confidence 70% or higher, use 'LongGrid'.
        - If the price prediction is 'Down' with confidence 70% or higher, use 'ShortGrid'.
        - For any other scenario (including confidence levels below 70%), use 'RegularGrid'.
        Ensure you adhere strictly to these confidence thresholds.
        End your response with a single word: 'RegularGrid', 'ShortGrid', or 'LongGrid'.""",
        expected_output="Recommended grid trading strategy with justification and final selection",
        agent=strategist
    )

    # Crew 인스턴스화
    crew = Crew(
        agents=[hourly_analyst, daily_analyst, price_predictor, strategist],
        tasks=[task1, task2, task3, task4],
        verbose=True,
        process=Process.sequential
    )

    results = crew.kickoff()
    print("CrewOutput type:", type(results))
    result_string = str(results)
    ##type: <class 'crewai.crews.crew_output.CrewOutput'>

    print("-----------------------------------------------------")
    print(results)  # results.raw)
    # print(results.raw)
    # print(results.tasks_output)
    # print(type(results.tasks_output))
    # <class 'list'>
    print("-----------------------------------------------------")

    task_results = {
        'hourly_analysis': results.tasks_output[0],
        'daily_analysis': results.tasks_output[1],
        'price_prediction': results.tasks_output[2],
        'strategy_recommendation': results.tasks_output[3]
    }

    price_prediction, confidence = extract_prediction(result_string)
    selected_strategy = extract_strategy(result_string)
    current_price = get_current_bitcoin_price(vt_symbol)

    # 한글 요약 생성 (이 부분은 그대로 유지)
    korean_summary_task = Task(
        description=f"""Summarize the following Bitcoin market analysis in Korean:
        1. Hourly Analysis: {task_results['hourly_analysis']}
        2. Daily Analysis: {task_results['daily_analysis']}
        3. Price Prediction: {task_results['price_prediction']}
        4. Strategy Recommendation: {task_results['strategy_recommendation']}

        현재 가격: {current_price}
        가격 예측: {price_prediction}
        신뢰도: {confidence}%
        선택된 전략: {selected_strategy}

        Provide a concise summary in Korean, highlighting the key points from each analysis and the final recommendations.
        Use natural Korean language and explain any technical terms if necessary.""",
        expected_output="A concise Korean summary of the Bitcoin market analysis and predictions",
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
        'korean_summary': korean_summary
    }
