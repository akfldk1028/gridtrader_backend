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
import requests

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
            now = datetime.now()
            next_run = now.replace(hour=9, minute=0, second=0, microsecond=0)

            # 만약 현재 시간이 오늘 오전 9시 이후라면, 다음 실행 시간을 오후 9시로 설정
            if now.hour >= 9:
                next_run = now.replace(hour=21, minute=0, second=0, microsecond=0)

            # 만약 현재 시간이 오후 9시 이후라면, 다음 날 오전 9시로 설정
            if now.hour >= 21:
                next_run = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)

            # now = datetime.now()
            # next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            # schedule(
            #     'binanaceAccount.tasks.trigger_save_daily_balance',
            #     schedule_type=Schedule.CRON,
            #     cron='0 */1 * * *',  # 매 3시간마다 정각에 실행
            #     next_run=next_hour,
            #     repeats=-1  # 무한 반복
            # )

            ##################################
            schedule(
                'binanaceAccount.tasks.trigger_save_daily_balance',
                schedule_type=Schedule.CRON,
                cron='0 9,0 * * *',  # 매일 오전 9시와 오후 9시에 실행
                next_run=next_run,
                repeats=-1  # 무한 반복
            )
        print(f"기록작업 {next_run.strftime('%Y-%m-%d %H:%M')}부터 하루에 두 번(오전 9시, 오후 9시) 실행되도록 예약되었습니다.")

    except ProgrammingError:
        # 데이터베이스 테이블이 아직 없는 경우
        print("Warning: Django-Q 테이블이 아직 생성되지 않았습니다. 마이그레이션을 실행해주세요.")
    except Exception as e:
        # 다른 예외 처리
        print(f"스케줄 설정 중 오류 발생: {str(e)}")


def get_future_account(viewName, max_retries=5, retry_delay=1):
    import time

    base_url = f"https://gridtrade.one/api/v1/binanaceAccount/{viewName}"


    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, timeout=60)
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


def trigger_save_daily_balance():
    print("Starting update_account_info task")
    try:
        futures_usdt_balance = get_future_account("get-future-balance")
        futures_positions = get_future_account("get-future-position")

        new_balance = DailyBalance.objects.create(
                futures_balance=futures_usdt_balance,
                futures_positions=futures_positions
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


