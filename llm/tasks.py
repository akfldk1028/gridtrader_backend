from .models import AnalysisResult
from .utils import perform_analysis  # 여기에 기존 분석 로직을 넣습니다.
import logging
from .models import AnalysisResult
from django_q.tasks import async_task, schedule
from django_q.models import Schedule
from datetime import time, datetime, timedelta
from TradeStrategy.models import StrategyConfig
from django.core.exceptions import ObjectDoesNotExist
import asyncio
from asgiref.sync import sync_to_async
from django.db import transaction


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
    next_hour = now.replace(hour=10, minute=20, second=0, microsecond=0)

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
        from .utils import perform_analysis
        result = await perform_analysis()

        analysis_result = await create_analysis_result({
            'symbol': result['symbol'],
            'result_string': result['result_string'],
            'current_price': result['current_price'],
            'price_prediction': result['price_prediction'],
            'confidence': float(result['confidence']) if result['confidence'] else None,
            'selected_strategy': result['selected_strategy']
        })

        await update_strategy_config(result['selected_strategy'])

        return f"Analysis completed successfully. AnalysisResult id: {analysis_result.id}"
    except Exception as e:
        logger.error(f"Error in run_bitcoin_analysis task: {str(e)}", exc_info=True)
        raise

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
        logger.info(f"Updated StrategyConfig grid_strategy to {selected_strategy}")
    except Exception as e:
        logger.error(f"Error updating StrategyConfig: {str(e)}", exc_info=True)


