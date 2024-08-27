from .models import AnalysisResult
from .utils import perform_analysis  # 여기에 기존 분석 로직을 넣습니다.
import logging
from .models import AnalysisResult
from django_q.tasks import async_task, schedule
from django_q.models import Schedule
from datetime import time, datetime, timedelta
from TradeStrategy.models import StrategyConfig
from django.core.exceptions import ObjectDoesNotExist

logger = logging.getLogger(__name__)


def setup_bitcoin_analysis_task():
    # 기존 스케줄이 있다면 삭제
    Schedule.objects.filter(func='llm.tasks.run_bitcoin_analysis').delete()

    # 현재 시간 가져오기
    now = datetime.now()

    # 다음 실행 시간 계산 (현재 시간의 다음 정각)
    next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    # 3시간마다 실행되는 스케줄 생성
    schedule(
        'llm.tasks.run_bitcoin_analysis',
        schedule_type=Schedule.HOURLY,
        next_run=next_run,
        repeats=-1,  # 무한 반복
        minutes=0,
        hours=3  # 3시간마다
    )

    # 즉시 한 번 실행
    async_task('llm.tasks.run_bitcoin_analysis')

    print(f"Bitcoin analysis task scheduled to run every 3 hours, starting from {next_run}")

# def setup_bitcoin_analysis_task():
#     # 기존 스케줄이 있다면 삭제
#     Schedule.objects.filter(func='llm.tasks.run_bitcoin_analysis').delete()
#
#     # 현재 날짜 가져오기
#     now = datetime.now()
#
#     # 오늘 오전 9시
#     morning_run = now.replace(hour=9, minute=0, second=0, microsecond=0)
#     if now > morning_run:
#         morning_run += timedelta(days=1)
#
#     # 오늘 오후 9시
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
#
#     # 오후 9시에 실행되는 스케줄 생성
#     schedule(
#         'llm.tasks.run_bitcoin_analysis',
#         schedule_type=Schedule.DAILY,
#         next_run=evening_run
#     )
#
#     # 즉시 한 번 실행
#     async_task('llm.tasks.run_bitcoin_analysis')



def update_strategy_config():
    try:
        # 가장 최근의 AnalysisResult 가져오기
        latest_analysis = AnalysisResult.objects.latest('created_at')

        if latest_analysis:
            selected_strategy = latest_analysis.selected_strategy

            # StrategyConfig 모델에서 설정 찾기 (이미지에서는 name이 '240824'입니다)
            strategy_config = StrategyConfig.objects.get(name='240824')

            # 현재 config 가져오기
            current_config = strategy_config.config

            # INIT 내의 setting의 grid_strategy 업데이트
            if 'INIT' in current_config and 'setting' in current_config['INIT']:
                current_config['INIT']['setting']['grid_strategy'] = selected_strategy

            # 업데이트된 config 저장
            strategy_config.config = current_config
            strategy_config.save()

            logger.info(f"Updated StrategyConfig grid_strategy to {selected_strategy}")
        else:
            logger.warning("No AnalysisResult found")

    except StrategyConfig.DoesNotExist:
        logger.error("StrategyConfig with name '240824' not found")
    except Exception as e:
        logger.error(f"Error updating StrategyConfig: {str(e)}", exc_info=True)


def run_bitcoin_analysis():
    logger.info("Starting Bitcoin analysis task")
    try:
        logger.info("Calling perform_analysis function")
        result = perform_analysis()
        logger.info(f"Analysis completed. Results: {result}")
        logger.info("Creating AnalysisResult object")
        #TODO SYMBOLE 별로 나중에 여러번
        analysis_result = AnalysisResult.objects.create(
            symbol=result['symbol'],
            result_string=result['result_string'],
            current_price=result['current_price'],
            price_prediction=result['price_prediction'],
            confidence=float(result['confidence']) if result['confidence'] else None,
            selected_strategy=result['selected_strategy']
        )

        logger.info(f"AnalysisResult object created with id: {analysis_result.id}")
        update_strategy_config()

        return f"Analysis completed successfully. AnalysisResult id: {analysis_result.id}"
    except Exception as e:
        logger.error(f"Error in run_bitcoin_analysis task: {str(e)}", exc_info=True)
        raise


def get_strategy_config(strategy_name='240824', update_grid_strategy=None):
    try:
        strategy_config = StrategyConfig.objects.get(name=strategy_name)
        config = strategy_config.config['INIT']

        vt_symbol = config.get('vt_symbol')
        symbol = vt_symbol.split('.')[0]  # "BNBUSDT.BINANCE"에서 "BNBUSDT" 추출
        grid_strategy = config['setting'].get('grid_strategy')

        if update_grid_strategy:
            config['setting']['grid_strategy'] = update_grid_strategy
            strategy_config.save()
            grid_strategy = update_grid_strategy

        return {
            'vt_symbol': symbol,
            'grid_strategy': grid_strategy
        }
    except ObjectDoesNotExist:
        print(f"Strategy configuration not found for: {strategy_name}")
        return None



# Django admin에서 주기적 태스크를 설정합니다.
# 1. Admin 페이지에 접속합니다.
# 2. "Periodic tasks" 섹션으로 이동합니다.
# 3. "ADD PERIODIC TASK" 버튼을 클릭합니다.
# 4. 태스크 이름을 입력합니다 (예: "Bitcoin Analysis").
# 5. Task (registered) 드롭다운에서 "analyzer.tasks.run_bitcoin_analysis"를 선택합니다.
# 6. Interval 스케줄을 선택하고 20분으로 설정합니다.
# 7. Save 버튼을 클릭합니다.

# 7. Celery Worker 및 Beat 실행

# 터미널에서 다음 명령어를 실행합니다.
# celery -A config worker -l info
# celery -A config beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler