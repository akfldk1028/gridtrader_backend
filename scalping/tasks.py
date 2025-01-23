from .models import TradingRecord  # 올바른 import
import logging
from django_q.tasks import async_task, schedule
from django_q.models import Schedule
from datetime import time, datetime, timedelta
import asyncio
from .utils import perform_analysis  # 여기에 기존 분석 로직을 넣습니다.
from .coin_selector import analyze_scalping_opportunities

logger = logging.getLogger(__name__)


def setup_scalping():
    # TODO 장기적관점엣 보는거 뭐 하루에한번하는걸로 한번 만들어야할듯? 뉴스만? 기술 1일 3일 결론도출해서 내 메인 AI에 음.. 한번씩만넣어야하나 ? 퍼센트를 나눠서?TASK
    # TASK1(초단기) 40 TASK2(중기) 40 TASK3(장기) 20 이런식으로?

    Schedule.objects.filter(func='scalping.tasks.scalping').delete()
    Schedule.objects.filter(func='scalping.tasks.scheduled_filter_and_save').delete()
    Schedule.objects.filter(func='scalping.tasks.binance_save').delete()
    Schedule.objects.filter(func='scalping.tasks.koreaStockSymbol').delete()
    Schedule.objects.filter(func='scalping.tasks.stockSymbol').delete()
    Schedule.objects.filter(func='scalping.tasks.SecondstockSymbol').delete()
    Schedule.objects.filter(func='scalping.tasks.ChinastockSymbol').delete()


    now = datetime.now()
    next_run = now + timedelta(minutes=60)

    # TASK1: 아침 8시 30분 및 오후 4시
    korea_morning_time = time(8, 30)
    korea_morning_datetime = datetime.combine(now.date(), korea_morning_time)
    # 다음 날로 시간 조정 (현재 시간이 이미 지정된 시간보다 늦었다면)
    if korea_morning_datetime < now:
        korea_morning_datetime += timedelta(days=1)

    # TASK1 스케줄 설정: 아침 8시 30분 및 오후 4시
    schedule(
        'scalping.tasks.koreaStockSymbol',
        schedule_type=Schedule.DAILY,
        next_run=korea_morning_datetime,
        repeats=-1  # 무한 반복
    )

    schedules = [
        {
            "task": "scalping.tasks.stockSymbol",
            "base_time": time(23, 0),  # 오후 11시
            "cron": "0 23/12 * * *"  # 오후 11시부터 12시간 간격
        },
        {
            "task": "scalping.tasks.SecondstockSymbol",
            "base_time": time(23, 30),  # 오후 11시 30분
            "cron": "30 23/12 * * *"  # 오후 11시 30분부터 12시간 간격
        },
        {
            "task": "scalping.tasks.ChinastockSymbol",
            "base_time": time(6, 30),  # 오전 6시 30분
            "cron": "30 6/12 * * *"  # 오전 6시 30분부터 12시간 간격
        }
    ]

    for schedule_info in schedules:
        task = schedule_info["task"]
        base_time = schedule_info["base_time"]
        cron = schedule_info["cron"]

        # 기존 스케줄 삭제
        Schedule.objects.filter(func=task).delete()

        # 다음 실행 시간 계산
        next_run = datetime.combine(now.date(), base_time)
        if now >= next_run:  # 기본 시간보다 현재 시간이 크면 다음 날로 이동
            next_run += timedelta(days=1)

        # 스케줄 등록 (기본 실행 + 12시간 반복)
        schedule(
            task,
            schedule_type=Schedule.CRON,
            cron=cron,
            next_run=next_run,
            repeats=-1  # 무한 반복
        )

    # TASK2: 저녁 11시 30분 및 오전 7시
    # stock_evening_time = time(23, 30)
    # stock_evening_datetime = datetime.combine(now.date(), stock_evening_time)
    #
    # if stock_evening_datetime < now:
    #     stock_evening_datetime += timedelta(days=1)
    # # TASK2 스케줄 설정: 저녁 11시 30분 및 오전 7시
    # schedule(
    #     'scalping.tasks.stockSymbol',
    #     schedule_type=Schedule.DAILY,
    #     next_run=stock_evening_datetime,
    #     repeats=-1  # 무한 반복
    # )










    schedule(
        'scalping.tasks.scheduled_filter_and_save',
        schedule_type=Schedule.MINUTES,  # MINUTES로 변경
        minutes=60,  # 1분마다 실행
        next_run=next_run,
        repeats=-1  # 무한 반복
    )


    schedule(
        'scalping.tasks.binance_save',
        schedule_type=Schedule.MINUTES,  # MINUTES로 변경
        minutes=60,  # 1분마다 실행
        next_run=next_run,
        repeats=-1  # 무한 반복
    )


    # schedule(
    #     'scalping.tasks.scalping',
    #     schedule_type=Schedule.MINUTES,  # MINUTES로 변경
    #     minutes=60,  # 1분마다 실행
    #     next_run=next_run,
    #     repeats=-1  # 무한 반복
    # )
    # schedule(
    #     'scalping.tasks.analyze_coins',  # 실행할 함수
    #     schedule_type=Schedule.MINUTES,   # 분 단위 실행
    #     minutes=10,                       # 10분마다 실행
    #     next_run=next_run,               # 다음 실행 시간
    #     repeats=-1                       # 무한 반복
    # )
def ChinastockSymbol():
    """Fetch Bitcoin data with technical indicators from API"""
    import requests

    base_url = "https://gridtrade.one/api/v1/binanceData/ChinaStockData/?all_last=true"
    session = requests.Session()
    for attempt in range(3):
        try:
            response = session.get(
                f"{base_url}",
                timeout=30,
                verify=False,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"데이터 가져오기 실패 (시도 {attempt + 1}/{3}): {e}")
            if attempt < 3 - 1:
                continue
        finally:
            session.close()

    return None

def SecondstockSymbol():
    """Fetch Bitcoin data with technical indicators from API"""
    import requests

    base_url = "https://gridtrade.one/api/v1/binanceData/stockData/?all_last_second=true"
    session = requests.Session()
    for attempt in range(3):
        try:
            response = session.get(
                f"{base_url}",
                timeout=30,
                verify=False,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"데이터 가져오기 실패 (시도 {attempt + 1}/{3}): {e}")
            if attempt < 3 - 1:
                continue
        finally:
            session.close()

    return None



def stockSymbol():
    """Fetch Bitcoin data with technical indicators from API"""
    import requests

    base_url = "https://gridtrade.one/api/v1/binanceData/stockData/?all_last=true"
    session = requests.Session()
    for attempt in range(3):
        try:
            response = session.get(
                f"{base_url}",
                timeout=30,
                verify=False,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"데이터 가져오기 실패 (시도 {attempt + 1}/{3}): {e}")
            if attempt < 3 - 1:
                continue
        finally:
            session.close()

    return None



def koreaStockSymbol():
    """Fetch Bitcoin data with technical indicators from API"""
    import requests

    base_url = "https://gridtrade.one/api/v1/binanceData/KoreaStockData/?all_last=true"
    session = requests.Session()
    for attempt in range(3):
        try:
            response = session.get(
                f"{base_url}",
                timeout=30,
                verify=False,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"데이터 가져오기 실패 (시도 {attempt + 1}/{3}): {e}")
            if attempt < 3 - 1:
              continue
        finally:
            session.close()

    return None


def binance_save():
    """Fetch Bitcoin data with technical indicators from API"""
    import requests

    base_url = "https://gridtrade.one/api/v1/binanceData/llm-bitcoin-data/?all_last=true"
    session = requests.Session()
    for attempt in range(3):
        try:
            response = session.get(
                f"{base_url}",
                timeout=30,
                verify=False,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"데이터 가져오기 실패 (시도 {attempt + 1}/{3}): {e}")
            if attempt < 3 - 1:
                continue
        finally:
            session.close()

    return None


def scheduled_filter_and_save():
    """Fetch Bitcoin data with technical indicators from API"""
    import requests

    base_url = "https://gridtrade.one/api/v1/binanceData/upbit/?all_last=true"
    session = requests.Session()
    for attempt in range(3):
        try:
            response = session.get(
                f"{base_url}",
                timeout=30,
                verify=False,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"데이터 가져오기 실패 (시도 {attempt + 1}/{3}): {e}")
            if attempt < 3 - 1:
                continue
        finally:
            session.close()

    return None


def scalping():
    try:
        print("Calling perform_analysis function")
        result = perform_analysis(symbol='KRW-DOGE')
        # result = perform_analysis(symbol='KRW-BTC')

        if result is None:
            print("Analysis failed▣▣▣▣▣▣▣▣▣▣▣")
        return f"Analysis completed successfully in seconds. AnalysisResult id: {result }"
    except Exception as e:
        print(f"Error in run_bitcoin_analysis task: {str(e)}")
        raise


def analyze_coins():
    """10분마다 실행되는 코인 분석 작업"""
    try:
        logger.info("Starting coin analysis task")
        results = analyze_scalping_opportunities()

        if results:
            coins = [result.coin_symbol for result in results]
            scores = [float(result.scalping_score) for result in results]
            logger.info(f"Analysis completed. Top coins: {', '.join(coins)}")
            return f"Successfully analyzed coins. Top recommendations: {', '.join(coins)}"
        else:
            logger.warning("No suitable coins found for scalping")
            return "Analysis completed but no suitable coins found"

    except Exception as e:
        logger.error(f"Error in coin analysis task: {str(e)}")
        raise