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
from django.apps import apps

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

            Schedule.objects.filter(func='binanaceAccount.tasks.trigger_save_daily_balance').delete()

            # now = datetime.now()
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            # next_hour = now.replace(hour=18, minute=35, second=0, microsecond=0)

            # 만약 현재 시간이 오늘 오전 9시 10분 이후라면, 다음 날로 설정
            # if now > next_hour:
            #     next_hour += timedelta(days=1)

            schedule(
                'binanaceAccount.tasks.trigger_save_daily_balance',
                schedule_type=Schedule.CRON,
                cron='0 */1 * * *',  # 매 3시간마다 정각에 실행
                next_run=next_hour,
                repeats=-1  # 무한 반복
            )
        print(f"기록작업 { next_hour.strftime('%Y-%m-%d %H:%M')}부터 1시간마다 실행되도록 예약되었습니다.")

    except ProgrammingError:
        # 데이터베이스 테이블이 아직 없는 경우
        print("Warning: Django-Q 테이블이 아직 생성되지 않았습니다. 마이그레이션을 실행해주세요.")
    except Exception as e:
        # 다른 예외 처리
        print(f"스케줄 설정 중 오류 발생: {str(e)}")


def trigger_save_daily_balance():
    print("Starting update_account_info task")
    factory = APIRequestFactory()
    request = factory.get('/')

    try:
        futures_balance_view = FuturesBalanceView()
        futures_balance_response = futures_balance_view.get(request)
        futures_usdt_balance = futures_balance_response.data


        futures_Position_view = FuturesPositionView()
        futures_Position_response = futures_Position_view.get(request)
        filtered_positions = futures_Position_response.data

        DailyBalance = apps.get_model('binanaceAccount', 'DailyBalance')

        # DailyBalance 모델에 저장
        new_balance = DailyBalance.objects.create(
                futures_balance=futures_usdt_balance,
                futures_positions=filtered_positions
            )
        print(
            f"Successfully created DailyBalance record with id {new_balance.id} at {new_balance.created_at}")

    except Exception as e:
        logger.error(f"Error saving daily balance: {str(e)}")
        print("Daily balance saved successfully");


def trigger_save_daily_balance_wrapper():
    trigger_save_daily_balance_websocket()

def trigger_save_daily_balance_websocket():
    print("trigger_save_daily_balance task started")
    channel_layer = get_channel_layer()
    try:
        async_to_sync(channel_layer.group_send)(
            "binanceQ",
            {
                "type": "save_daily_balance",
            }
        )
        print("save_daily_balance message sent to channel layer")
    except Exception as e:
        print(f"Error in trigger_save_daily_balance: {str(e)}")



def set_cache_data(account_type, key, data):
    """
    Redis에 데이터를 UTF-8로 인코딩하여 저장하는 함수
    """
    cache_key = f"{account_type}:{key}"
    encoded_data = json.dumps(data).encode('utf-8')
    caches['account'].set(cache_key, encoded_data, timeout=CACHE_TIMEOUT)



# 캐시에서 데이터를 가져오는 함수 (필요시 사용)
def get_cache_data(account_type, key):
    cache_key = f"{account_type}:{key}"
    encoded_data = caches['account'].get(cache_key)
    if encoded_data:
        return json.loads(encoded_data.decode('utf-8'))
    return None


