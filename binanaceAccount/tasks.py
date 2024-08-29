import logging
from django.core.cache import caches
from .views import SpotAccountInfoView, FuturesAccountInfoView, SpotBalanceView, FuturesBalanceView,FuturesPositionView
from rest_framework.test import APIRequestFactory
import json
from datetime import time, datetime, timedelta
from TradeLog.views import BaseDataView  # BaseDataView를 TradeLog 앱에서 가져옵니다
from django_q.tasks import schedule, async_task
from django_q.models import Schedule
from django.db.utils import ProgrammingError
from django.db import transaction
from .models import DailyBalance
from django.utils import timezone
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import websockets
import json
import asyncio

logger = logging.getLogger(__name__)
CACHE_TIMEOUT = 60 * 3  # 10 minutes in seconds


def setup_update_account_info_task():
    try:
        with transaction.atomic():
            # 기존 스케줄이 있다면 삭제
            # Schedule.objects.filter(func='binanaceAccount.tasks.update_account_info').delete()
            # schedule(
            #     'binanaceAccount.tasks.update_account_info',
            #     schedule_type='I',
            #     minutes=2,
            #     repeats=-1
            # )

            Schedule.objects.filter(func='binanaceAccount.tasks.trigger_save_daily_balance_wrapper').delete()

            # now = datetime.now()
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            # next_hour = now.replace(hour=11, minute=45, second=0, microsecond=0)

            # 만약 현재 시간이 오늘 오전 9시 10분 이후라면, 다음 날로 설정
            # if now > next_hour:
            #     next_hour += timedelta(days=1)

            schedule(
                'binanaceAccount.tasks.trigger_save_daily_balance_wrapper',
                schedule_type=Schedule.CRON,
                cron='0 */1 * * *',  # 매 3시간마다 정각에 실행
                next_run=next_hour,
                repeats=-1  # 무한 반복
            )

    except ProgrammingError:
        # 데이터베이스 테이블이 아직 없는 경우
        print("Warning: Django-Q 테이블이 아직 생성되지 않았습니다. 마이그레이션을 실행해주세요.")
    except Exception as e:
        # 다른 예외 처리
        print(f"스케줄 설정 중 오류 발생: {str(e)}")


async def trigger_save_daily_balance():
    print("Starting update_account_info task")
    uri = f"wss://gridtrader-backend.onrender.com/ws/binanceQ/"

    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({
                'action': 'save_daily_balance'
            }))

            response = await websocket.recv()
            response_data = json.loads(response)

            if response_data.get('type') == 'daily_balance_saved':
                print("Daily balance saved successfully")
            else:
                print(f"Error: {response_data.get('message')}")

    except Exception as e:
        print(f"Error saving daily balance: {str(e)}")

def trigger_save_daily_balance_wrapper():
    asyncio.run(trigger_save_daily_balance())

def set_cache_data(account_type, key, data):
    """
    Redis에 데이터를 UTF-8로 인코딩하여 저장하는 함수
    """
    cache_key = f"{account_type}:{key}"
    encoded_data = json.dumps(data).encode('utf-8')
    caches['account'].set(cache_key, encoded_data, timeout=CACHE_TIMEOUT)




def update_account_info():
    print("Starting update_account_info task")
    factory = APIRequestFactory()
    request = factory.get('/')

    try:
        # Spot Account Info
        logger.info("Fetching Spot Account Info")
        spot_account_view = SpotAccountInfoView()
        spot_account_response = spot_account_view.get(request)
        spot_account_info = spot_account_response.data
        # logger.info(f"Spot Account Info: {spot_account_info}")
        set_cache_data('spot', 'account_info', spot_account_info)

        # Futures Account Info
        logger.info("Fetching Futures Account Info")
        futures_account_view = FuturesAccountInfoView()
        futures_account_response = futures_account_view.get(request)
        futures_account_info = futures_account_response.data
        # logger.info(f"Futures Account Info: {futures_account_info}")
        set_cache_data('futures', 'account_info', futures_account_info)

        # Spot Balance
        logger.info("Fetching Spot Balance")
        spot_balance_view = SpotBalanceView()
        spot_balance_response = spot_balance_view.get(request)
        spot_balances = spot_balance_response.data
        # logger.info(f"Spot Balances: {spot_balances}")
        set_cache_data('spot', 'balances', spot_balances)

        # Futures Balance
        logger.info("Fetching Futures Balance")
        futures_balance_view = FuturesBalanceView()
        futures_balance_response = futures_balance_view.get(request)
        futures_usdt_balance = futures_balance_response.data
        # logger.info(f"Futures USDT Balance: {futures_usdt_balance}")
        set_cache_data('futures', 'usdt_balance', futures_usdt_balance)


        # Futures Position
        logger.info("Fetching Positions")
        futures_Position_view = FuturesPositionView()
        futures_Position_response = futures_Position_view.get(request)
        futures = futures_Position_response.data
        # logger.info(f"Futures USDT Balance: {futures_usdt_balance}")
        set_cache_data('futures', 'positions', futures)


        logger.info("Account info successfully updated in cache")
        return "Update completed successfully"
    except Exception as e:
        logger.error(f"Error updating account info: {str(e)}")
        raise


# 캐시에서 데이터를 가져오는 함수 (필요시 사용)
def get_cache_data(account_type, key):
    cache_key = f"{account_type}:{key}"
    encoded_data = caches['account'].get(cache_key)
    if encoded_data:
        return json.loads(encoded_data.decode('utf-8'))
    return None


def simple_cache_test(self):
    logger.info(f"Task id: {self.request.id}")
    logger.info("Starting simple_cache_test")
    try:
        account_cache = caches['account']
        logger.info(f"Cache backend: {account_cache.backend.__class__.__name__}")  # Redis 백엔드 확인
        account_cache.set('test_key', 'test_value', timeout=None)
        logger.info("Cache set completed")
        verify_value = account_cache.get('test_key')
        logger.info(f"Verified value from cache: {verify_value}")
        return f"Cache test completed. Verified value: {verify_value}"
    except Exception as e:
        logger.error(f"Error in simple_cache_test: {str(e)}")
        raise