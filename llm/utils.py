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
from rest_framework.test import APIRequestFactory
from binanceData.views import BinanceLLMChartDataAPIView
from rest_framework.response import Response

# Binance 클라이언트 초기화
symbol = "BTCUSDT"

binance_api_key = settings.BINANCE_API_KEY
binance_api_secret = settings.BINANCE_API_SECRET
client = Client(binance_api_key, binance_api_secret)
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
        description=f"""Conduct a comprehensive analysis of the Bitcoin market using the most recent 120 hours of hourly data:
        {bitcoin_data['hourly'][-120:]}

        IMPORTANT: Start your analysis from the most recent data point (the last entry in the provided dataset) and work backwards.

        Focus on the most recent 120 hours, examining:
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
        description="""Based on the detailed hourly and daily analyses provided, predict future Bitcoin price scenarios for the next 6-24 hours (short-term) and 1-3 days (long-term):

        1. Describe one bullish and one bearish scenario for each timeframe
        2. Include specific price targets or ranges for each scenario
        3. Identify immediate potential triggers or catalysts for each scenario, referencing the technical analysis from the hourly and daily analyses
        4. Assign probabilities to each scenario (ensure they sum to 100% per timeframe)
        5. Highlight key technical levels to watch in the very near term, as identified in the previous analyses
    
        IMPORTANT: Focus on synthesizing the information from the hourly and daily analyses to forecast immediate future developments, emphasizing short-term trading perspectives. Carefully consider the following points, then provide probabilities for each scenario and explain your reasoning in detail:
        The most significant technical indicators and patterns identified in the hourly and daily analyses (After consideration, Probability: X%, Reason: ...)
        Potential rapid market sentiment shifts based on the analyzed trends (After consideration, Probability: Y%, Reason: ...)
        Immediate changes in trading patterns and volume as highlighted in the previous analyses (After consideration, Probability: Z%, Reason: ...)
        
        IMPORTANT: Ensure that your probability assessments and explanations reflect a thorough consideration of all available information from both the hourly and daily analyses.
                
        Based on your synthesis of the previous analyses, provide a single most likely direction for the next 6-48 hours.
        
        End your response with either 'Up' or 'Down' followed by the confidence percentage, e.g., 'Up 80%', 'Down 85%', 'Up 70%' or 'Down 65%'. 
        Ensure that your confidence level reflects the strength and consistency of the indicators across both timeframes.""",
        expected_output="Concise short-term future scenario analysis for Bitcoin with a single directional prediction and confidence level, based on the synthesis of hourly and daily technical analyses",
        agent=price_predictor
    )

    task4 = Task(
        description="""Based on the market analyses provided for both 1-hour and 1-day timeframes, and considering the price prediction,
        determine the most suitable grid trading strategy among regular grid, short grid, and long grid. 
        Provide a clear rationale for your choice, considering both short-term and long-term market conditions.
        Use the following guidelines:

        1. Analyze market trend:
           - Examine both short-term (1-hour) and long-term (1-day) trends.
           - If both trends align strongly (either upward or downward), proceed to step 2.
           - If trends are conflicting or unclear, lean towards 'RegularGrid'.

        2. Evaluate confidence level and technical indicators:
           - If the price prediction is 'Up' with confidence 70% or higher, and technical indicators support an uptrend, consider 'LongGrid'.
           - If the price prediction is 'Down' with confidence 70% or higher, and technical indicators support a downtrend, consider 'ShortGrid'.
           - If confidence is below 70% or technical indicators are mixed, use 'RegularGrid'.

        3. Consider market momentum and volume:
           - Strong upward momentum with increasing volume supports 'LongGrid'.
           - Strong downward momentum with increasing volume supports 'ShortGrid'.
           - Weak momentum or inconsistent volume suggests 'RegularGrid'.

        4. Final decision:
           - Choose 'LongGrid' or 'ShortGrid' if market trend, confidence level, technical indicators, and momentum all align strongly.
           - Choose 'RegularGrid' if there's any significant contradiction among these factors or if the market direction is uncertain.

        Provide a brief explanation for your choice, referencing the above criteria.
        End your response with a single word: 'RegularGrid', 'ShortGrid', or 'LongGrid'.""",
        expected_output="Recommended grid trading strategy with justification and final selection, balancing trend analysis and technical indicators while maintaining the 70% confidence threshold",
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

    selected_strategy = extract_strategy(result_string)
    price_prediction, confidence = extract_prediction(selected_strategy, result_string)
    # current_price = get_current_bitcoin_price(vt_symbol)
    current_price = 0
    # 한글 요약 생성 (이 부분은 그대로 유지)
    # Korean summary generation (keep this part as is)

    korean_summary_task = Task(
        description=f"""Summarize the following Bitcoin market analysis in Korean:
        1. Hourly Analysis: {task_results['hourly_analysis']}
        2. Daily Analysis: {task_results['daily_analysis']}
        3. Price Prediction: {task_results['price_prediction']}
        4. Strategy Recommendation: {task_results['strategy_recommendation']}

        Price Prediction: {price_prediction}
        Confidence: {confidence}%

        Provide a detailed summary in Korean, highlighting the key points from each analysis. Explain any technical terms if necessary.
        The hourly analysis, daily analysis, Price Prediction and Probability Assessments must be analyzed and presented separately in detail.

        Translate the final conclusion and selected strategy as follows:

        ★ Final Conclusion: {result_string}
        ★ Selected Strategy: {selected_strategy}

       IMPORTANT: Structure your response clearly and elegantly using the following format:

        1. Use Markdown headers (##) for each main section: 시간별 분석, 일별 분석, 가격 예측, 확률 평가, 전략 추천, 주요 지표, 최종 결론, 선택된 전략.
        2. Use bullet points or numbered lists for key points within each section.
        3. Highlight important information using bold text or symbols.
        4. Present the 주요 지표 (Key Indicators) section as a list with clear labels.
        5. Use the ★ symbol before the Final Conclusion and Selected Strategy.
        6. Add a horizontal rule (---) after each section to clearly separate them.

        Ensure that your summary is easy to read at a glance, with clear separation between sections and emphasis on crucial information.""",
        expected_output="A well-structured, clear, and concise Korean summary of the Bitcoin market analysis and predictions, with translated final conclusion and selected strategy, formatted for easy readability and clear section separation",
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
