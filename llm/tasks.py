from .models import AnalysisResult
import logging
from django_q.tasks import async_task, schedule
from django_q.models import Schedule
from datetime import time, datetime, timedelta
from TradeStrategy.models import StrategyConfig
from .utils import perform_analysis
from .utilsTrade import perform_new_analysis
from .ETH import perform_eth_analysis

from asgiref.sync import sync_to_async
import asyncio

logger = logging.getLogger(__name__)


def setup_bitcoin_analysis_task():
    # TODO 장기적관점엣 보는거 뭐 하루에한번하는걸로 한번 만들어야할듯? 뉴스만? 기술 1일 3일 결론도출해서 내 메인 AI에 음.. 한번씩만넣어야하나 ? 퍼센트를 나눠서?TASK
    # TASK1(초단기) 40 TASK2(중기) 40 TASK3(장기) 20 이런식으로?

    # Schedule.objects.filter(func='llm.tasks.run_bitcoin_analysis').delete()
    # Schedule.objects.filter(func='llm.tasks.run_eth_analysis').delete()
    Schedule.objects.filter(func='llm.tasks.run_new_bitcoin_analysis').delete()

    now = datetime.now()
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=2)

    # 다음 실행 시간을 오전 9시 10분으로 설정
    # next_hour = now.replace(hour=16, minute=55, second=0, microsecond=0)
    if now > next_hour:
        next_hour += timedelta(days=1)


    schedule(
        'llm.tasks.run_new_bitcoin_analysis',
        schedule_type=Schedule.CRON,
        cron="35 2 * * *",  # 매일 오전 3시, 오전 9시, 오후 3시, 오후 10시에 실행
        next_run=next_hour,
        repeats=-1  # 무한 반복
    )
    # schedule(
    #     'llm.tasks.run_eth_analysis',
    #     schedule_type=Schedule.CRON,
    #     cron="45 0,4,8,12,16,20 * * *",  # 매일 오전 3시, 오전 9시, 오후 3시, 오후 10시에 실행
    #     next_run=next_eth_hour,
    #     repeats=-1  # 무한 반복
    # )
    # # async_task('llm.tasks.run_bitcoin_analysis')
    # print(f"분석 작업이 {next_hour.strftime('%Y-%m-%d %H:%M')}부터 3시간마다 실행되도록 예약되었습니다.")
    #

def update_strategy_config():
    try:
        # StrategyConfig 모델에서 설정 찾기
        strategy_config = StrategyConfig.objects.get(name='240824')

        # 현재 config 가져오기
        current_config = strategy_config.config

        # 각 Symbol별로 최신 분석 결과 가져오기
        symbols = ['BTCUSDT', 'ETHUSDT']
        for symbol in symbols:
            latest_analysis = AnalysisResult.objects.filter(symbol=symbol).order_by('-created_at').first()

            if latest_analysis:
                selected_strategy = latest_analysis.selected_strategy

                # Symbol에 따라 적절한 설정 업데이트
                if symbol == 'BTCUSDT':
                    if 'INIT' in current_config and 'setting' in current_config['INIT']:
                        current_config['INIT']['setting']['grid_strategy'] = selected_strategy
                elif symbol == 'ETHUSDT':
                    if 'ETH' in current_config and 'setting' in current_config['ETH']:
                        current_config['ETH']['setting']['grid_strategy'] = selected_strategy

                logger.info(f"Updated StrategyConfig grid_strategy for {symbol} to {selected_strategy}")
            else:
                logger.warning(f"No AnalysisResult found for {symbol}")

        # 업데이트된 config 저장
        strategy_config.config = current_config
        strategy_config.save()

    except StrategyConfig.DoesNotExist:
        logger.error("StrategyConfig with name '240824' not found")
    except Exception as e:
        logger.error(f"Error updating StrategyConfig: {str(e)}")




def run_new_bitcoin_analysis():
    print("Starting Bitcoin analysis task")
    try:
        print("Calling perform_analysis function")
        result = perform_new_analysis()
        # result = asyncio.run(perform_analysis())
        if result is None:
            print("Analysis failed▣▣▣▣▣▣▣▣▣▣▣")

        return f"Analysis completed successfully in seconds."
    except Exception as e:
        print(f"Error in run_bitcoin_analysis task: {str(e)}")
        raise




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
            korean_summary=result['korean_summary'] if 'korean_summary' in result else "",
            analysis_results_30m=result['analysis_results_30m'],
            analysis_results_1hour=result['analysis_results_1hour'],
            analysis_results_daily=result['analysis_results_daily'],
        )
        update_strategy_config()
        return f"Analysis completed successfully in seconds. AnalysisResult id: {analysis_result.id}"
    except Exception as e:
        print(f"Error in run_bitcoin_analysis task: {str(e)}")
        raise


def run_eth_analysis():
    print("Starting ETH analysis task")
    try:
        print("Calling perform_eth_analysis function")
        result = perform_eth_analysis()

        if result is None:
            print("ETH Analysis failed: perform_eth_analysis returned None")
            logger.error("ETH Analysis failed: perform_eth_analysis returned None")
            return "ETH Analysis failed: No result returned"

        print("Creating AnalysisResult object")
        analysis_result = AnalysisResult.objects.create(
            symbol=result.get('symbol', 'ETHUSDT'),  # Default to 'ETHUSDT' if not present
            result_string=result.get('result_string', ''),
            current_price=result.get('current_price', 0),
            price_prediction=result.get('price_prediction', ''),
            confidence=float(result['confidence']) if result.get('confidence') else None,
            selected_strategy=result.get('selected_strategy', ''),
            korean_summary=result.get('korean_summary', ''),
            analysis_results_30m=result.get('analysis_results_30m', ''),
            analysis_results_1hour=result.get('analysis_results_1hour', ''),
            analysis_results_daily=result.get('analysis_results_daily', ''),
        )
        update_strategy_config()
        return f"ETH Analysis completed successfully. AnalysisResult id: {analysis_result.id}"
    except Exception as e:
        error_message = f"Error in run_eth_analysis task: {str(e)}"
        print(error_message)
        logger.error(error_message)
        return error_message
