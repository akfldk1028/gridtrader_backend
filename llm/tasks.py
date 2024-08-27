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

    # 다음 실행 시간을 오늘 또는 내일 12시 20분으로 설정
    next_run = now.replace(hour=12, minute=40, second=0, microsecond=0)
    if now >= next_run:
        next_run += timedelta(days=1)

    # 3시간마다 실행되는 스케줄 생성 (CRON 사용)
    schedule(
        'llm.tasks.run_bitcoin_analysis',
        schedule_type=Schedule.CRON,
        cron='20 */3 * * *',  # 매 3시간마다 20분에 실행
        next_run=next_run,
        repeats=-1  # 무한 반복
    )

    # 즉시 한 번 실행
    # async_task('llm.tasks.run_bitcoin_analysis')

    print(f"비트코인 분석 작업이 {next_run.strftime('%Y-%m-%d %H:%M')}부터 3시간마다 실행되도록 예약되었습니다.")

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
    import time

    start_time = time.time()
    print("Starting Bitcoin analysis task")
    try:
        print("Calling perform_analysis function")
        analysis_start = time.time()
        result = perform_analysis()
        analysis_end = time.time()
        print(f"Analysis completed in {analysis_end - analysis_start:.2f} seconds. Results: {result}")

        print("Creating AnalysisResult object")
        db_start = time.time()
        analysis_result = AnalysisResult.objects.create(
            symbol=result['symbol'],
            result_string=result['result_string'],
            current_price=result['current_price'],
            price_prediction=result['price_prediction'],
            confidence=float(result['confidence']) if result['confidence'] else None,
            selected_strategy=result['selected_strategy']
        )
        db_end = time.time()
        print(f"AnalysisResult object created in {db_end - db_start:.2f} seconds. ID: {analysis_result.id}")

        update_start = time.time()
        update_strategy_config()
        update_end = time.time()
        print(f"Strategy config updated in {update_end - update_start:.2f} seconds")

        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f} seconds")

        return f"Analysis completed successfully in {total_time:.2f} seconds. AnalysisResult id: {analysis_result.id}"
    except Exception as e:
        print(f"Error in run_bitcoin_analysis task: {str(e)}")
        raise


# def get_strategy_config(strategy_name='240824', update_grid_strategy=None):
#     try:
#         strategy_config = StrategyConfig.objects.get(name=strategy_name)
#         config = strategy_config.config['INIT']
#
#         vt_symbol = config.get('vt_symbol')
#         symbol = vt_symbol.split('.')[0]  # "BNBUSDT.BINANCE"에서 "BNBUSDT" 추출
#         grid_strategy = config['setting'].get('grid_strategy')
#
#         if update_grid_strategy:
#             config['setting']['grid_strategy'] = update_grid_strategy
#             strategy_config.save()
#             grid_strategy = update_grid_strategy
#
#         return {
#             'vt_symbol': symbol,
#             'grid_strategy': grid_strategy
#         }
#     except ObjectDoesNotExist:
#         print(f"Strategy configuration not found for: {strategy_name}")
#         return None
#
