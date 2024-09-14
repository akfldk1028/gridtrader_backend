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
symbol = "BTCUSDT"

# binance_api_key = settings.BINANCE_API_KEY
# binance_api_secret = settings.BINANCE_API_SECRET
# client = Client(binance_api_key, binance_api_secret)
cryptocompare_api_key = "400daae3cf09044e5d78b3fc744b107731547031372de5573431166b96d16db7"


def get_crypto_news():
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={cryptocompare_api_key}"
    response = requests.get(url)
    news_data = response.json()['Data']
    return news_data[:30]  # 최근 10개의 뉴스만 반환


# from langchain.chat_models import ChatOpenAI
# # from langchain_openai import ChatOpenAI
# # 파이썬버젼이 딸려서
# custom_llm = ChatOpenAI(
#     model="gpt-4o-mini",  # 또는 원하는 다른 모델
#     temperature=0.7,
#     max_tokens=16000,
# )

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
    role=f'15-Minute {symbol} Market Analyst',
    goal='Analyze Bitcoin market trends and patterns in 15-minute timeframe',
    backstory="""You are an experienced cryptocurrency market analyst specializing in very short-term Bitcoin analysis.
    Your expertise lies in technical analysis and identifying market trends in 15-minute charts.
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
    You pay close attention to market trends, recognizing their significant impact on short-term price movements and adjusting your strategies accordingly.
    Your trend analysis includes identifying key support and resistance levels, recognizing trend reversals, and adapting to different market phases (trending, ranging, or volatile).
    While profit maximization remains a priority, you never compromise on protecting your capital from liquidation events.
    Your ability to align your trades with the prevailing market trend while remaining vigilant to potential trend shifts sets you apart as a trader.""",
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


# def get_current_bitcoin_price(vt_symbol):
#     try:
#         ticker = client.get_symbol_ticker(symbol=vt_symbol)
#         return float(ticker['price'])
#     except BinanceAPIException as e:
#         print(f"Error fetching current Bitcoin price: {e}")
#         return None
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


# def get_bitcoin_data_from_api(symbol, max_retries=5, retry_delay=1):
#     # APIRequestFactory 사용
#     factory = APIRequestFactory()
#     request = factory.get('/', {'symbol': symbol})
#     try:
#         # 뷰 클래스의 인스턴스 생성
#         LLMChartData = BinanceLLMChartDataAPIView()
#         response = LLMChartData.get(request)
#
#         # Response 객체 확인
#         if isinstance(response, Response):
#             # DRF Response 객체인 경우
#             if response.status_code == 200:
#                 return response.data
#             else:
#                 raise Exception(f"Error: {response.status_code}, {response.data}")
#         else:
#             # 일반 Django HttpResponse 객체인 경우 (드문 경우)
#             if response.status_code == 200:
#                 import json
#                 return json.loads(response.content)
#             else:
#                 raise Exception(f"Error: {response.status_code}, {response.content}")
#
#     except Exception as e:
#         print(f"Error occurred while fetching Binance chart data: {e}")
#         return None

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

    # task_news = Task(
    #     description=f"""Analyze the following recent cryptocurrency news items and determine their potential impact on Bitcoin's price trend:
    #     {[{
    #         'title': news['title'],
    #         'content': news['body'],
    #     } for news in get_crypto_news()]}
    #
    #     1. Quickly review all news items, focusing on their potential impact on Bitcoin's price.
    #     2. Identify key themes or events that could significantly influence Bitcoin's price.
    #     3. Based on the overall sentiment of the news, determine:
    #        a) The likely short-term trend (next 24-48 hours): Bullish, Bearish, or Neutral
    #        b) The potential long-term trend (next 1-2 weeks): Bullish, Bearish, or Neutral
    #
    #     Provide a concise summary (3-5 sentences) explaining your trend predictions, referencing the most impactful news items.
    #
    #     End your analysis with two lines:
    #     "NEWS_Short-term trend: [Bullish/Bearish/Neutral]"
    #     "NEWS_Long-term trend: [Bullish/Bearish/Neutral]"
    #     """,
    #     expected_output="Concise analysis of Bitcoin's likely price trends based on recent news, with clear short-term and long-term trend predictions.",
    #     agent=news_analyst
    # )

    task_30min = Task(
        description=f"""Analyze the Bitcoin market using the latest 48 hours of 30-minute data:
        {bitcoin_data['30min'][-96:]}

        IMPORTANT: Start from the most recent data point and analyze backwards.

        Focus on:
        1. **Price Trends & Formations**: Identify upward or downward trends and any emerging trend patterns.
        2. **Volume Patterns**: Detect volume changes and their correlation with price movements.
        3. **Ichimoku Cloud Indicators**:
            - **Tenkan-sen & Kijun-sen Crossovers**: Identify bullish (Tenkan > Kijun) or bearish (Tenkan < Kijun) signals.
            - **Price vs. Senkou Span A & B**: Determine if the price is above or below the cloud.
            - **Chikou Span Position**: Check if Chikou Span is above or below the current price.
        4. **Stochastic Oscillator**: Note %K and %D crossovers indicating potential trend reversals.
        5. **RSI Divergences**: Highlight overbought (>70) or oversold (<30) conditions and any divergences.

        Compare the 30-minute trends with the 15-minute analysis to identify confirmations or divergences.

        Conclude with:
        - **Market Sentiment**: Bullish, Bearish, or Neutral based on the indicators.
        - **Short-term Outlook**: 4-8 hours.

        Ensure clarity and focus on key indicators to accurately determine market trends.""",
        expected_output="Concise and accurate Bitcoin market analysis report for the 30-minute timeframe, emphasizing key Ichimoku Cloud signals and other technical indicators.",
        agent=thirty_min_analyst
    )

    # 태스크 생성
    task1 = Task(
        description=f"""Analyze the Bitcoin market using the latest 120 hours of hourly data:
        {bitcoin_data['hourly'][-120:]}

        IMPORTANT: Start from the most recent data point and analyze backwards.

        Focus on:
        1. **Price Trends**: Identify upward or downward trends starting from the latest price.
        2. **Volume Changes**: Compare recent volumes with earlier ones to gauge market interest.
        3. **Ichimoku Cloud Indicators**:
            - **Kumo (Cloud) Position**: Is the price above or below the cloud?
            - **Tenkan-sen vs. Kijun-sen**: Look for crossovers (Bullish or Bearish signals).
            - **Chikou Span**: Position relative to the current price.
        4. **RSI (Relative Strength Index)**: Highlight overbought (>70) or oversold (<30) conditions.
        5. **Stochastic Oscillator**: Note %K and %D crossovers indicating potential trend reversals.
        6. **Technical Patterns**: Detect formations like Head and Shoulders, Double Tops/Bottoms, etc.

        Conclude with:
        - **Market Sentiment**: Bullish, Bearish, or Neutral based on the above indicators.
        - **Short-term Outlook**: 6-12 hours.
        - **Long-term Outlook**: 12-24 hours.

        Ensure clarity and focus on key indicators to accurately determine market trends.""",
        expected_output="Concise and accurate Bitcoin market analysis report for the 1-hour timeframe, emphasizing key Ichimoku Cloud signals and other technical indicators.",
        agent=hourly_analyst
    )

    task2 = Task(
        description=f"""Analyze the Bitcoin market using the latest 90 days of daily data:
        {bitcoin_data['daily'][-90:]}

        IMPORTANT: Start from the most recent data point and analyze backwards.

        Focus on:
        1. **Price Trends & Key Levels**: Identify upward or downward trends and significant support/resistance levels.
        2. **Volume Patterns**: Detect significant volume spikes and their correlation with price movements.
        3. **Ichimoku Cloud Indicators**:
            - **Price vs. Kumo (Cloud)**: Determine if the price is above or below the cloud.
            - **Tenkan-sen & Kijun-sen Crossovers**: Identify bullish (Tenkan > Kijun) or bearish (Tenkan < Kijun) signals.
            - **Chikou Span Position**: Check if Chikou Span is above or below the current price.
        4. **RSI (Relative Strength Index)**: Highlight overbought (>70) or oversold (<30) conditions.
        5. **Stochastic Oscillator**: Note %K and %D crossovers indicating potential trend reversals.
        6. **Technical Patterns**: Identify formations like Head and Shoulders, Double Tops/Bottoms, Triangles, Flags, etc.

        Conclude with:
        - **Market Sentiment**: Bullish, Bearish, or Neutral based on the indicators.
        - **Short-term Outlook**: 12-24 hours.
        - **Long-term Outlook**: 1-3 days.

        Ensure clarity and focus on key indicators to accurately determine market trends.""",
        expected_output="Concise and accurate Bitcoin market analysis report for the 1-day timeframe, emphasizing key Ichimoku Cloud signals and other technical indicators.",
        agent=daily_analyst
    )

    task3 = Task(
        description="""Based on the analyses from the **30-minute and 1-hour timeframes**, forecast Bitcoin price movements for the next:

    1. **1-6 hours (Very Short-term)**
    2. **6-24 hours (Short-term)**
    3. **1-3 days (Medium-term)**

    For each timeframe, provide:

    1. **Single Direction Prediction**: Determine the most likely direction (Up/Down) using primarily the 30-minute and 1-hour analyses.
    2. **Confidence Level**: Assign a confidence percentage based on the strength and agreement of indicators from these short-term timeframes, **adjusted slightly** by the overall trend from the daily timeframe.
    3. **Key Technical Levels**: Highlight crucial support and resistance levels.

    **Guidelines:**

    - **Focus on Short-Term Timeframes**: Use the 30-minute and 1-hour analyses as the primary basis for all predictions.
    - **Use Daily Timeframe for Confidence Adjustment Only**: Refer to the daily timeframe solely to adjust the confidence level, not to influence the prediction direction.
        - **If the daily trend aligns with the short-term prediction**, slightly increase the confidence level.
        - **If the daily trend opposes the short-term prediction**, slightly decrease the confidence level.
    - **Do Not Base Predictions on Daily Timeframe**: Predictions should be made based on short-term analyses regardless of the daily trend.
    - **Conciseness**: Keep predictions clear and to the point.

    **Output Format:**

    End your response with three lines indicating the predicted direction and confidence level for each timeframe:
         "1-6 hours: [Up/Down] [Confidence]%"
         "6-24 hours: [Up/Down] [Confidence]%"
         "1-3 days: [Up/Down] [Confidence]%"
    """,
        expected_output="Accurate Bitcoin price predictions with directional outcomes and confidence levels, based primarily on short-term timeframes, with confidence adjusted by the daily trend.",
        agent=price_predictor
    )

    task4 = Task(
        description="""Determine the most suitable grid trading strategy (**RegularGrid**, **ShortGrid**, **LongGrid**) for Bitcoin based on the predictions from the **30-minute and 1-hour timeframes**, including Ichimoku Cloud signals and technical indicators.

    **Guidelines:**

    1. **Strategy Selection Criteria:**

       - **LongGrid**:
            - Select if both the 30-minute and 1-hour predictions indicate 'Up' with a confidence level of **70% or higher**.
       - **ShortGrid**:
            - Select if both the 30-minute and 1-hour predictions indicate 'Down' with a confidence level of **70% or higher**.
       - **RegularGrid**:
            - Select if the predictions are mixed or the confidence levels are below 70%.

    2. **Handling Conflicting Signals:**

       - **Compare Confidence Levels**:
         - When the daily timeframe conflicts with the short-term predictions, compare the confidence levels of both.
       - **Adjust Confidence Levels**:
         - Slightly adjust the overall confidence to reflect any uncertainty caused by conflicting signals.

    3. **Maximize Profit:**

       - Choose the strategy that best leverages the predictions with the higher confidence level to maximize profit.
       - Be responsive to the market signals with the strongest indications.

    4. **Conciseness:**

       - Provide only the strategy name without additional explanation.

    **Decision Rules:**

    - **Select LongGrid** if the conditions favor an upward trend with higher confidence.
    - **Select ShortGrid** if the conditions favor a downward trend with higher confidence.
    - **Select RegularGrid** if predictions are mixed or confidence levels do not clearly favor one direction.

    **Output Format:**

    At the end of your response, provide a single word: 'RegularGrid', 'ShortGrid', or 'LongGrid'.
    """,
        expected_output="""Recommend a grid trading strategy based on the predictions with higher confidence levels, appropriately handling conflicting signals by comparing confidence levels between timeframes.""",
        agent=strategist
    )

    # Crew 인스턴스화
    crew = Crew(
        agents=[thirty_min_analyst, hourly_analyst, daily_analyst, price_predictor, strategist],
        tasks=[task_30min, task1, task2, task3, task4],
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
        '30min_analysis': results.tasks_output[0],
        'hourly_analysis': results.tasks_output[1],
        'daily_analysis': results.tasks_output[2],
        'price_prediction': results.tasks_output[3],
        'strategy_recommendation': results.tasks_output[4]
    }

    selected_strategy = extract_strategy(result_string)
    price_prediction, confidence = extract_prediction(selected_strategy, result_string)
    # current_price = get_current_bitcoin_price(vt_symbol)
    current_price = 0
    # 한글 요약 생성 (이 부분은 그대로 유지)
    # Korean summary generation (keep this part as is)

    korean_summary_task = Task(
        description=f"""Summarize the following Bitcoin market analysis in Korean:
        1. 30-Minute Analysis: {task_results['30min_analysis']}
        2. Hourly Analysis: {task_results['hourly_analysis']}
        3. Daily Analysis: {task_results['daily_analysis']}
        4. Price Prediction: {task_results['price_prediction']}
        5. Strategy Recommendation: {task_results['strategy_recommendation']}

        Price Prediction: {price_prediction}
        Confidence: {confidence}%

        Provide a detailed summary in Korean, highlighting the key points from each analysis. Explain any technical terms if necessary.
        The 30-minute, hourly, and daily analyses, as well as the Price Prediction and Probability Assessments must be analyzed and presented separately in detail.

        **Additionally, include a summary of the Ichimoku Cloud analysis based on the data from the 30-minute, hourly, and daily analyses.**

        Translate the final conclusion and selected strategy as follows:

        ★ Final Conclusion: {result_string}
        ★ Selected Strategy: {selected_strategy}

       IMPORTANT: Structure your response clearly and elegantly using the following format:

        1. Use Markdown headers (##) for each main section:
           - 30분 분석
           - 시간별 분석
           - 일별 분석
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
        'korean_summary': korean_summary
    }
