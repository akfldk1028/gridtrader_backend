# import os
# from crewai import Agent, Task, Crew, Process
# from binance.client import Client
# from binance.exceptions import BinanceAPIException
# import pandas as pd
# import numpy as np
# import re
# from TradeStrategy.models import StrategyConfig
# from django.conf import settings
# from django.core.exceptions import ObjectDoesNotExist
# import websockets
# import json
# from asgiref.sync import sync_to_async
# from django.db import transaction
#
# # Binance 클라이언트 초기화
# symbol = "BNBUSDT"
# binance_api_key = settings.BINANCE_API_KEY
# binance_api_secret = settings.BINANCE_API_SECRET
# client = Client(binance_api_key, binance_api_secret)
#
# hourly_analyst = Agent(
#     role='Hourly Bitcoin Market Analyst',
#     goal='Analyze Bitcoin market trends and patterns in 1-hour timeframe',
#     backstory="""You are an experienced cryptocurrency market analyst specializing in short-term Bitcoin analysis.
#     Your expertise lies in technical analysis and identifying market trends in hourly charts.
#     You are known for your balanced and objective analysis, considering both bullish and bearish scenarios.""",
#     verbose=True,
#     allow_delegation=False,
# )
#
# daily_analyst = Agent(
#     role='Daily Bitcoin Market Analyst',
#     goal='Analyze Bitcoin market trends and patterns in 1-day timeframe',
#     backstory="""You are an experienced cryptocurrency market analyst specializing in medium to long-term Bitcoin analysis.
#     Your expertise lies in technical analysis and identifying market trends in daily charts.
#     You are known for your cautious approach, always considering multiple market scenarios.""",
#     verbose=True,
#     allow_delegation=False,
# )
#
# strategist = Agent(
#     role='Grid Trading Strategist',
#     goal='Determine the most suitable grid trading strategy based on market analysis',
#     backstory="""You are a seasoned trading strategist with deep knowledge of various grid trading techniques.
#     You excel at matching market conditions with appropriate trading strategies.
#     You are known for your adaptive approach, often recommending a mix of strategies or regular grid trading in uncertain markets.""",
#     verbose=True,
#     allow_delegation=False,
# )
#
# price_predictor = Agent(
#     role='Bitcoin Price Predictor',
#     goal='Predict the future price movement of Bitcoin and provide a confidence level',
#     backstory="""You are an expert in price prediction for cryptocurrencies, especially Bitcoin.
#     You use a combination of technical analysis, market sentiment, and historical patterns to make educated guesses about future price movements.
#     You are known for your conservative estimates and rarely give extremely high confidence levels.""",
#     verbose=True,
#     allow_delegation=False,
# )
#
# def extract_prediction(text):
#     # Look for the specific format: "Up X%" or "Down X%"
#     match = re.search(r'(Up|Down)\s+(\d+(?:\.\d+)?)%', text, re.IGNORECASE)
#     if match:
#         return match.group(1), match.group(2)
#
#     # If not found, look for any mention of "Up" or "Down" near a percentage
#     up_match = re.search(r'Up.*?(\d+(?:\.\d+)?)%', text, re.IGNORECASE)
#     down_match = re.search(r'Down.*?(\d+(?:\.\d+)?)%', text, re.IGNORECASE)
#
#     if up_match:
#         return "Up", up_match.group(1)
#     elif down_match:
#         return "Down", down_match.group(1)
#
#     return None, None
#
# def extract_strategy(text):
#     strategies = ["RegularGrid", "ShortGrid", "LongGrid"]
#     for strategy in strategies:
#         if strategy in text:
#             return strategy
#     return None
#
#
#
# @sync_to_async
# def get_strategy_config(strategy_name='240824'):
#     try:
#         strategy_config = StrategyConfig.objects.get(name=strategy_name)
#         first_config = strategy_config.config['INIT']
#
#         vt_symbol = first_config.get('vt_symbol')
#         symbol = vt_symbol.split('.')[0] if vt_symbol else ''  # "BNBUSDT.BINANCE"에서 "BNBUSDT" 추출
#         print("--시발호출")
#         print(symbol)
#         grid_strategy = first_config['setting'].get('grid_strategy')
#
#         if not symbol or not grid_strategy:
#             raise ValueError("Invalid configuration: missing symbol or grid_strategy")
#
#         return {
#             'vt_symbol': symbol,
#             'grid_strategy': grid_strategy
#         }
#     except ObjectDoesNotExist:
#         print(f"Strategy configuration not found for: {strategy_name}")
#     except Exception as e:
#         print(f"Error in get_strategy_config: {str(e)}")
#     return None
#
#
# async def perform_analysis():
#     try:
#         config = await get_strategy_config()
#         if not config:
#             print("Strategy configuration is invalid.")
#             return None
#
#         vt_symbol = config['vt_symbol']
#         grid_strategy = config['grid_strategy']
#
#         print(f"Current grid_strategy: {grid_strategy}")
#
#         uri = "wss://gridtrader-backend.onrender.com/ws/binanceQ/"
#         async with websockets.connect(uri) as websocket:
#             # Bitcoin 데이터 요청
#             await websocket.send(json.dumps({
#                 'action': 'get_bitcoin_data_and_price',
#                 'symbol': vt_symbol
#             }))
#
#             # 응답 대기
#             response = await websocket.recv()
#             data = json.loads(response)
#             print(data)
#             if data.get('type') != 'llm_data_and_price':
#                 print(f"Unexpected response type: {data.get('type')}")
#                 return None
#
#             bitcoin_data = data.get('data', {})
#             if not bitcoin_data:
#                 print("Failed to receive Bitcoin data")
#                 return None
#
#             print("Received Bitcoin data:", bitcoin_data)
#
#             current_price = bitcoin_data.get('current_price')
#             print(f'current_price {current_price}')
#             if current_price is None:
#                 current_price = 0
#
#
#             # 태스크 생성
#             task1 = Task(
#                 description=f"""Conduct a comprehensive analysis of the Bitcoin market using the provided hourly data:
#                 {bitcoin_data.get('hourly', [])}
#                 Examine price trends, volume, RSI, and Stochastic oscillator.
#                 Identify significant support and resistance levels, and overall market sentiment in the 1-hour timeframe.
#                 Consider both bullish and bearish scenarios in your analysis.""",
#                 expected_output="Detailed Bitcoin market analysis report for 1-hour timeframe",
#                 agent=hourly_analyst
#             )
#
#             task2 = Task(
#                 description=f"""Conduct a comprehensive analysis of the Bitcoin market using the provided daily data:
#                 {bitcoin_data.get('daily', [])}
#                 Examine price trends, volume, RSI, and Stochastic oscillator.
#                 Identify significant support and resistance levels, and overall market sentiment in the 1-day timeframe.
#                 Consider both bullish and bearish scenarios in your analysis.""",
#                 expected_output="Detailed Bitcoin market analysis report for 1-day timeframe",
#                 agent=daily_analyst
#             )
#
#             task3 = Task(
#                 description="""Based on all the analyses provided, predict whether the Bitcoin price is more likely to go up or down in the near future.
#                 Provide a brief explanation for your prediction and assign a confidence level to your prediction as a percentage.
#                 Look at the short-term and long-term situation and evaluate it objectively. If you make a mistake, your current Bitcoin futures investment may be liquidated.
#                 End your response with either 'Up' or 'Down' followed by the confidence percentage, e.g., 'Up 70%' or 'Down 65%'.""",
#                 expected_output="Bitcoin price movement prediction with explanation and confidence level",
#                 agent=price_predictor
#             )
#
#             task4 = Task(
#                 description="""Based on the market analyses provided for both 1-hour and 1-day timeframes, and considering the price prediction,
#                 determine the most suitable grid trading strategy among regular grid, short grid, and long grid.
#                 Provide a clear rationale for your choice, considering both short-term and long-term market conditions.
#                 Use the following guidelines, but also consider the overall market analysis:
#                 - If the price prediction is 'Up' with confidence over 70% or 70%, consider 'LongGrid'.
#                 - If the price prediction is 'Down' with confidence over 70% or 70%, consider 'ShortGrid'.
#                 - For confidence levels between 55-69%, consider a mix of strategies or lean towards 'RegularGrid'.
#                 - For confidence levels below 55%, strongly consider 'RegularGrid'.
#                 End your response with a single word: 'RegularGrid', 'ShortGrid', or 'LongGrid'.""",
#                 expected_output="Recommended grid trading strategy with justification and final selection",
#                 agent=strategist
#             )
#
#             # Crew 인스턴스화
#             crew = Crew(
#                 agents=[hourly_analyst, daily_analyst, price_predictor, strategist],
#                 tasks=[task1, task2, task3, task4],
#                 verbose=True,
#                 process=Process.sequential
#             )
#
#             result = crew.kickoff()
#             result_string = str(result)
#
#             price_prediction, confidence = extract_prediction(result_string)
#             selected_strategy = extract_strategy(result_string)
#             print("######################")
#             print("간다이이이이이잇")
#             print(f"Analysis complete. Results have been saved to report.md and the database.")
#             print(f"Selected Grid Strategy: {selected_strategy}")
#             print(f"Price Prediction: {price_prediction}")
#             print(f"Confidence Level: {confidence}%")
#
#             # 결과 생성 및 저장
#             analysis_result = {
#                 'symbol': vt_symbol,
#                 'result_string': result_string,
#                 'current_price': current_price,
#                 'price_prediction': price_prediction,
#                 'confidence': float(confidence) if confidence else None,
#                 'selected_strategy': selected_strategy
#             }
#
#             # 결과를 데이터베이스에 저장
#             saved_result = await create_analysis_result(analysis_result)
#             await update_strategy_config(analysis_result['selected_strategy'])
#
#             return saved_result
#
#     except websockets.exceptions.WebSocketException as e:
#         print(f"WebSocket error: {str(e)}")
#     except json.JSONDecodeError as e:
#         print(f"JSON decoding error: {str(e)}")
#     except Exception as e:
#         print(f"Unexpected error in perform_analysis: {str(e)}")
#
#     return None
#
#
# @sync_to_async
# def create_analysis_result(data):
#     from .models import AnalysisResult
#     with transaction.atomic():
#         return AnalysisResult.objects.create(**data)
#
#
# @sync_to_async
# def update_strategy_config(selected_strategy):
#     try:
#         with transaction.atomic():
#             strategy_config = StrategyConfig.objects.get(name='240824')
#             current_config = strategy_config.config
#             if 'INIT' in current_config and 'setting' in current_config['INIT']:
#                 current_config['INIT']['setting']['grid_strategy'] = selected_strategy
#             strategy_config.config = current_config
#             strategy_config.save()
#         print(f"Updated StrategyConfig grid_strategy to {selected_strategy}")
#     except Exception as e:
#         print(f"Error updating StrategyConfig: {str(e)}")


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

# Binance 클라이언트 초기화
symbol = "BNBUSDT"

binance_api_key = settings.BINANCE_API_KEY
binance_api_secret = settings.BINANCE_API_SECRET
client = Client(binance_api_key, binance_api_secret)


def get_bitcoin_data(symbol):
    try:
        # 1시간 및 1일 간격의 데이터 가져오기
        hourly_candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=500)
        daily_candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY, limit=500)

        def process_candles(candles):
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                                'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(
                float)

            # RSI 계산
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # 스토캐스틱 계산
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['%K'] = (df['close'] - low_14) / (high_14 - low_14) * 100
            df['%D'] = df['%K'].rolling(window=3).mean()

            return df.to_dict(orient='records')

        return {
            'hourly': process_candles(hourly_candles),
            'daily': process_candles(daily_candles)
        }
    except BinanceAPIException as e:
        print(f"An error occurred: {e}")
        return None


hourly_analyst = Agent(
    role='Hourly Bitcoin Market Analyst',
    goal='Analyze Bitcoin market trends and patterns in 1-hour timeframe',
    backstory="""You are an experienced cryptocurrency market analyst specializing in short-term Bitcoin analysis.
    Your expertise lies in technical analysis and identifying market trends in hourly charts.
    You are known for your balanced and objective analysis, considering both bullish and bearish scenarios.""",
    verbose=True,
    allow_delegation=False,
)

daily_analyst = Agent(
    role='Daily Bitcoin Market Analyst',
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
    role='Bitcoin Price Predictor',
    goal='Predict the future price movement of Bitcoin and provide a confidence level',
    backstory="""You are an expert in price prediction for cryptocurrencies, especially Bitcoin. 
    You use a combination of technical analysis, market sentiment, and historical patterns to make educated guesses about future price movements.
    You are known for your conservative estimates and rarely give extremely high confidence levels.""",
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
def extract_prediction(text):
    # Look for the specific format: "Up X%" or "Down X%"
    match = re.search(r'(Up|Down)\s+(\d+(?:\.\d+)?)%', text, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)

    # If not found, look for any mention of "Up" or "Down" near a percentage
    up_match = re.search(r'Up.*?(\d+(?:\.\d+)?)%', text, re.IGNORECASE)
    down_match = re.search(r'Down.*?(\d+(?:\.\d+)?)%', text, re.IGNORECASE)

    if up_match:
        return "Up", up_match.group(1)
    elif down_match:
        return "Down", down_match.group(1)

    return None, None


def extract_strategy(text):
    strategies = ["RegularGrid", "ShortGrid", "LongGrid"]
    for strategy in strategies:
        if strategy in text:
            return strategy
    return None


def get_current_bitcoin_price():
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
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


# 기존 get_current_bitcoin_price 함수 내용...

def perform_analysis():
    config = get_strategy_config()
    if not config:
        print("Strategy configuration is invalid.")
        return None

    vt_symbol = config['vt_symbol']
    grid_strategy = config['grid_strategy']

    print(f"Current grid_strategy: {grid_strategy}")

    bitcoin_data = get_bitcoin_data(vt_symbol)
    if not bitcoin_data:
        print("Failed to fetch bitcoin data.")
        return None

    # 태스크 생성
    task1 = Task(
        description=f"""Conduct a comprehensive analysis of the Bitcoin market using the provided hourly data:
        {bitcoin_data['hourly']}
        Examine price trends, volume, RSI, and Stochastic oscillator. 
        Identify significant support and resistance levels, and overall market sentiment in the 1-hour timeframe.
        Consider both bullish and bearish scenarios in your analysis.""",
        expected_output="Detailed Bitcoin market analysis report for 1-hour timeframe",
        agent=hourly_analyst
    )

    task2 = Task(
        description=f"""Conduct a comprehensive analysis of the Bitcoin market using the provided daily data:
        {bitcoin_data['daily']}
        Examine price trends, volume, RSI, and Stochastic oscillator. 
        Identify significant support and resistance levels, and overall market sentiment in the 1-day timeframe.
        Consider both bullish and bearish scenarios in your analysis.""",
        expected_output="Detailed Bitcoin market analysis report for 1-day timeframe",
        agent=daily_analyst
    )

    task3 = Task(
        description="""Based on all the analyses provided, predict whether the Bitcoin price is more likely to go up or down in the near future.
        Provide a brief explanation for your prediction and assign a confidence level to your prediction as a percentage.
        Look at the short-term and long-term situation and evaluate it objectively. If you make a mistake, your current Bitcoin futures investment may be liquidated.
        End your response with either 'Up' or 'Down' followed by the confidence percentage, e.g., 'Up 70%' or 'Down 65%'.""",
        expected_output="Bitcoin price movement prediction with explanation and confidence level",
        agent=price_predictor
    )

    task4 = Task(
        description="""Based on the market analyses provided for both 1-hour and 1-day timeframes, and considering the price prediction,
        determine the most suitable grid trading strategy among regular grid, short grid, and long grid. 
        Provide a clear rationale for your choice, considering both short-term and long-term market conditions.
        Use the following guidelines, but also consider the overall market analysis:
        - If the price prediction is 'Up' with confidence over 70% or 70%, consider 'LongGrid'.
        - If the price prediction is 'Down' with confidence over 70% or 70%, consider 'ShortGrid'.
        - For confidence levels between 55-69%, consider a mix of strategies or lean towards 'RegularGrid'.
        - For confidence levels below 55%, strongly consider 'RegularGrid'.
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
    result_parts = result_string.split("\n\n")
    print("-----------------------------------------------------")
    print(results)

    task_results = {
        'hourly_analysis': result_parts[0] if len(result_parts) > 0 else "",
        'daily_analysis': result_parts[1] if len(result_parts) > 1 else "",
        'price_prediction': result_parts[2] if len(result_parts) > 2 else "",
        'strategy_recommendation': result_parts[3] if len(result_parts) > 3 else ""
    }
    # result_parts = result.split("\n\n")  # 각 태스크의 결과는 빈 줄로 구분되어 있다고 가정

    # price_prediction, confidence = extract_prediction(result_string)
    # selected_strategy = extract_strategy(task_results['strategy_recommendation'])
    # current_price = get_current_bitcoin_price()
    price_prediction, confidence = extract_prediction(task_results['price_prediction'])
    selected_strategy = extract_strategy(task_results['strategy_recommendation'])
    current_price = get_current_bitcoin_price()


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
