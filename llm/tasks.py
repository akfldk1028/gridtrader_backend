from .models import AnalysisResult
import logging
from django_q.tasks import async_task, schedule
from django_q.models import Schedule
from datetime import time, datetime, timedelta
import asyncio
from django.db import transaction
from crewai import Agent, Task, Crew, Process
from binance.client import Client
import re
from asgiref.sync import sync_to_async
from TradeStrategy.models import StrategyConfig
from django.core.exceptions import ObjectDoesNotExist
import websockets
import json

logger = logging.getLogger(__name__)


def setup_bitcoin_analysis_task():
    # 기존 스케줄이 있다면 삭제
    Schedule.objects.filter(func='llm.tasks.run_bitcoin_analysis_sync').delete()


    #     evening_run = now.replace(hour=21, minute=0, second=0, microsecond=0)
    #     if now > evening_run:
    #         evening_run += timedelta(days=1)
    #
    #     # 오전 9시에 실행되는 스케줄 생성
    #     schedule(
    #         'llm.tasks.run_bitcoin_analysis',
    #         schedule_type=Schedule.DAILY,
    #         next_run=morning_run
    #     )
    # 현재 시간 가져오기


    now = datetime.now()
    # next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


    # 다음 실행 시간을 오전 9시 10분으로 설정
    next_hour = now.replace(hour=16, minute=8, second=0, microsecond=0)

    # 만약 현재 시간이 오늘 오전 9시 10분 이후라면, 다음 날로 설정
    if now > next_hour:
        next_hour += timedelta(days=1)


    # 작업을 정각에 실행하고, 그 후에는 3시간마다 반복
    schedule(
        'llm.tasks.run_bitcoin_analysis_sync',
        schedule_type=Schedule.CRON,
        cron='0 */3 * * *',  # 매 3시간마다 정각에 실행
        next_run=next_hour,
        repeats=-1  # 무한 반복
    )

    # 즉시 한 번 실행
    # async_task('llm.tasks.run_bitcoin_analysis')
    ########print
    print('시시시시ㅣ발')
    print(f"비트코인 분석 작업이 { next_hour.strftime('%Y-%m-%d %H:%M')}부터 3시간마다 실행되도록 예약되었습니다.")


def run_bitcoin_analysis_sync():
    return asyncio.run(run_bitcoin_analysis())

async def run_bitcoin_analysis():
    try:
        # from .utils import perform_analysis
        result = await perform_analysis()
        print(result)
    except Exception as e:
        logger.error(f"Error in run_bitcoin_analysis task: {str(e)}", exc_info=True)
        raise






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



@sync_to_async
def get_strategy_config(strategy_name='240824'):
    try:
        strategy_config = StrategyConfig.objects.get(name=strategy_name)
        first_config = strategy_config.config['INIT']

        vt_symbol = first_config.get('vt_symbol')
        symbol = vt_symbol.split('.')[0] if vt_symbol else ''  # "BNBUSDT.BINANCE"에서 "BNBUSDT" 추출
        print("--시발호출")
        print(symbol)
        grid_strategy = first_config['setting'].get('grid_strategy')

        if not symbol or not grid_strategy:
            raise ValueError("Invalid configuration: missing symbol or grid_strategy")

        return {
            'vt_symbol': symbol,
            'grid_strategy': grid_strategy
        }
    except ObjectDoesNotExist:
        print(f"Strategy configuration not found for: {strategy_name}")
    except Exception as e:
        print(f"Error in get_strategy_config: {str(e)}")
    return None


async def perform_analysis():
    try:
        config = await get_strategy_config()
        if not config:
            print("Strategy configuration is invalid.")
            return None

        vt_symbol = config['vt_symbol']
        grid_strategy = config['grid_strategy']

        print(f"Current grid_strategy: {grid_strategy}")

        uri = "wss://gridtrader-backend.onrender.com/ws/binanceQ/"
        async with websockets.connect(uri) as websocket:
            # Bitcoin 데이터 요청
            await websocket.send(json.dumps({
                'action': 'get_bitcoin_data_and_price',
                'symbol': vt_symbol
            }))

            # 응답 대기
            response = await websocket.recv()
            data = json.loads(response)
            print(data)
            if data.get('type') != 'llm_data_and_price':
                print(f"Unexpected response type: {data.get('type')}")
                return None

            bitcoin_data = data.get('data', {})
            if not bitcoin_data:
                print("Failed to receive Bitcoin data")
                return None

            print("Received Bitcoin data:", bitcoin_data)

            current_price = bitcoin_data.get('current_price')
            print(f'current_price {current_price}')
            if current_price is None:
                current_price = 0


            # 태스크 생성
            task1 = Task(
                description=f"""Conduct a comprehensive analysis of the Bitcoin market using the provided hourly data:
                {bitcoin_data.get('hourly', [])}
                Examine price trends, volume, RSI, and Stochastic oscillator. 
                Identify significant support and resistance levels, and overall market sentiment in the 1-hour timeframe.
                Consider both bullish and bearish scenarios in your analysis.""",
                expected_output="Detailed Bitcoin market analysis report for 1-hour timeframe",
                agent=hourly_analyst
            )

            task2 = Task(
                description=f"""Conduct a comprehensive analysis of the Bitcoin market using the provided daily data:
                {bitcoin_data.get('daily', [])}
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

            result = crew.kickoff()
            result_string = str(result)

            price_prediction, confidence = extract_prediction(result_string)
            selected_strategy = extract_strategy(result_string)
            print("######################")
            print("간다이이이이이잇")
            print(f"Analysis complete. Results have been saved to report.md and the database.")
            print(f"Selected Grid Strategy: {selected_strategy}")
            print(f"Price Prediction: {price_prediction}")
            print(f"Confidence Level: {confidence}%")

            # 결과 생성 및 저장
            analysis_result = {
                'symbol': vt_symbol,
                'result_string': result_string,
                'current_price': current_price,
                'price_prediction': price_prediction,
                'confidence': float(confidence) if confidence else None,
                'selected_strategy': selected_strategy
            }

            # 결과를 데이터베이스에 저장
            saved_result = await create_analysis_result(analysis_result)
            await update_strategy_config(analysis_result['selected_strategy'])

            return saved_result

    except websockets.exceptions.WebSocketException as e:
        print(f"WebSocket error: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in perform_analysis: {str(e)}")

    return None


@sync_to_async
def create_analysis_result(data):
    from .models import AnalysisResult
    with transaction.atomic():
        return AnalysisResult.objects.create(**data)


@sync_to_async
def update_strategy_config(selected_strategy):
    try:
        with transaction.atomic():
            strategy_config = StrategyConfig.objects.get(name='240824')
            current_config = strategy_config.config
            if 'INIT' in current_config and 'setting' in current_config['INIT']:
                current_config['INIT']['setting']['grid_strategy'] = selected_strategy
            strategy_config.config = current_config
            strategy_config.save()
        print(f"Updated StrategyConfig grid_strategy to {selected_strategy}")
    except Exception as e:
        print(f"Error updating StrategyConfig: {str(e)}")