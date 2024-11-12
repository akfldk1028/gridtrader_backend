from .models import AnalysisResult
import logging
from django_q.tasks import async_task, schedule
from django_q.models import Schedule
from datetime import time, datetime, timedelta
import asyncio
from .utils import perform_analysis  # 여기에 기존 분석 로직을 넣습니다.

logger = logging.getLogger(__name__)


def setup_scalping():
    # TODO 장기적관점엣 보는거 뭐 하루에한번하는걸로 한번 만들어야할듯? 뉴스만? 기술 1일 3일 결론도출해서 내 메인 AI에 음.. 한번씩만넣어야하나 ? 퍼센트를 나눠서?TASK
    # TASK1(초단기) 40 TASK2(중기) 40 TASK3(장기) 20 이런식으로?

    Schedule.objects.filter(func='scalping.tasks.scalping').delete()
    now = datetime.now()
    next_run = now + timedelta(minutes=1)


    schedule(
        'scalping.tasks.scalping',
        schedule_type=Schedule.MINUTES,  # MINUTES로 변경
        minutes=1,  # 1분마다 실행
        next_run=next_run,
        repeats=-1  # 무한 반복
    )

def scalping():
    try:
        print("Calling perform_analysis function")
        result = perform_analysis()
        if result is None:
            print("Analysis failed▣▣▣▣▣▣▣▣▣▣▣")
        return f"Analysis completed successfully in seconds. AnalysisResult id: {result }"
    except Exception as e:
        print(f"Error in run_bitcoin_analysis task: {str(e)}")
        raise
