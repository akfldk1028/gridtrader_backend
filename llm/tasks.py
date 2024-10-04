from .models import AnalysisResult
import logging
from django_q.tasks import async_task, schedule
from django_q.models import Schedule
from datetime import time, datetime, timedelta
from TradeStrategy.models import StrategyConfig
from .utils import perform_analysis  # 여기에 기존 분석 로직을 넣습니다.
from .ETH import perform_eth_analysis  # 여기에 기존 분석 로직을 넣습니다.

from asgiref.sync import sync_to_async
import asyncio

logger = logging.getLogger(__name__)


def setup_bitcoin_analysis_task():
    #TODO 장기적관점엣 보는거 뭐 하루에한번하는걸로 한번 만들어야할듯? 뉴스만? 기술 1일 3일 결론도출해서 내 메인 AI에 음.. 한번씩만넣어야하나 ? 퍼센트를 나눠서?TASK
    # TASK1(초단기) 40 TASK2(중기) 40 TASK3(장기) 20 이런식으로?

    Schedule.objects.filter(func='llm.tasks.run_bitcoin_analysis').delete()
    now = datetime.now()
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


    # 다음 실행 시간을 오전 9시 10분으로 설정
    # next_hour = now.replace(hour=16, minute=55, second=0, microsecond=0)

    # 만약 현재 시간이 오늘 오전 9시 10분 이후라면, 다음 날로 설정
    if now > next_hour:
        next_hour += timedelta(days=1)


    # 작업을 정각에 실행하고, 그 후에는 3시간마다 반복
    # schedule(
    #     'llm.tasks.run_bitcoin_analysis',
    #     schedule_type=Schedule.CRON,
    #     cron='0 */3 * * *',  # 매 3시간마다 정각에 실행
    #     next_run=next_hour,
    #     repeats=-1  # 무한 반복
    # )
    #     cron="0 3,8,11,15,19,22 * * *",  # 매일 오전 3시, 오전 9시, 오후 3시, 오후 10시에 실행
    # 10분정도에 걸어놔야할듯?
    schedule(
        'llm.tasks.run_bitcoin_analysis',
        schedule_type=Schedule.CRON,
        cron="0 8,11,15,20,23 * * *",  # 매일 오전 3시, 오전 9시, 오후 3시, 오후 10시에 실행
        next_run=next_hour,
        repeats=-1  # 무한 반복
    )
    schedule(
        'llm.tasks.run_eth_analysis',
        schedule_type=Schedule.CRON,
        cron="5 8,11,15,20,23 * * *",  # 매일 오전 3시, 오전 9시, 오후 3시, 오후 10시에 실행
        next_run=next_hour,
        repeats=-1  # 무한 반복
    )
    # async_task('llm.tasks.run_bitcoin_analysis')
    print(f"분석 작업이 { next_hour.strftime('%Y-%m-%d %H:%M')}부터 3시간마다 실행되도록 예약되었습니다.")



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
            elif 'ETH' in current_config and 'setting' in current_config['ETH']:
                current_config['ETH']['setting']['grid_strategy'] = selected_strategy

            # 업데이트된 config 저장
            strategy_config.config = current_config
            strategy_config.save()

            logger.info(f"Updated StrategyConfig grid_strategy to {selected_strategy}")
        else:
            logger.warning("No AnalysisResult found")

    except StrategyConfig.DoesNotExist:
        logger.error("StrategyConfig with name '240824' not found")
    except Exception as e:
        logger.error(f"Error updating StrategyConfig: {str(e)}")


# async def run_bitcoin_analysis_async():
#     print("Starting Bitcoin analysis task")
#     try:
#         print("Calling perform_analysis function")
#         result = await perform_analysis()
#         print("Creating AnalysisResult object")
#         analysis_result = await sync_to_async(AnalysisResult.objects.create)(
#             symbol=result['symbol'],
#             result_string=result['result_string'],
#             current_price=result['current_price'],
#             price_prediction=result['price_prediction'],
#             confidence=float(result['confidence']) if result['confidence'] else None,
#             selected_strategy=result['selected_strategy'],
#             korean_summary = result['korean_summary'] if 'korean_summary' in result else ""
#         )
#         await sync_to_async(update_strategy_config)()
#         return f"Analysis completed successfully in seconds. AnalysisResult id: {analysis_result.id}"
#     except Exception as e:
#         print(f"Error in run_bitcoin_analysis task: {str(e)}")
#         raise



def run_bitcoin_analysis():
    print("Starting Bitcoin analysis task")
    try:
        print("Calling perform_analysis function")
        result = perform_analysis()
        # result = asyncio.run(perform_analysis())
        if result is None:
            print("Analysis failed▣▣▣▣▣▣▣▣▣▣▣")

        print("Creating AnalysisResult object")
        analysis_result = AnalysisResult.objects.create(
            symbol=result['symbol'],
            result_string=result['result_string'],
            current_price=result['current_price'],
            price_prediction=result['price_prediction'],
            confidence=float(result['confidence']) if result['confidence'] else None,
            selected_strategy=result['selected_strategy'],
            korean_summary = result['korean_summary'] if 'korean_summary' in result else "",
            analysis_results_30m = result['analysis_results_30m'],
            analysis_results_1hour = result['analysis_results_1hour'],
            analysis_results_daily = result['analysis_results_daily'],
        )
        update_strategy_config()
        return f"Analysis completed successfully in seconds. AnalysisResult id: {analysis_result.id}"
    except Exception as e:
        print(f"Error in run_bitcoin_analysis task: {str(e)}")
        raise


def run_eth_analysis():
    print("Starting ETH analysis task")
    try:
        print("Calling perform_analysis function")
        result = perform_eth_analysis()
        # result = asyncio.run(perform_analysis())
        if result is None:
            print("Analysis failed▣▣▣▣▣▣▣▣▣▣▣")

        print("Creating AnalysisResult object")
        analysis_result = AnalysisResult.objects.create(
            symbol=result['symbol'],
            result_string=result['result_string'],
            current_price=result['current_price'],
            price_prediction=result['price_prediction'],
            confidence=float(result['confidence']) if result['confidence'] else None,
            selected_strategy=result['selected_strategy'],
            korean_summary = result['korean_summary'] if 'korean_summary' in result else "",
            analysis_results_30m = result['analysis_results_30m'],
            analysis_results_1hour = result['analysis_results_1hour'],
            analysis_results_daily = result['analysis_results_daily'],
        )
        update_strategy_config()
        return f"Analysis completed successfully in seconds. AnalysisResult id: {analysis_result.id}"
    except Exception as e:
        print(f"Error in run_bitcoin_analysis task: {str(e)}")
        raise