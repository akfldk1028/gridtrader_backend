from django.apps import AppConfig
from django.db.models.signals import post_migrate
from django.db.utils import OperationalError, ProgrammingError

class LlmConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'llm'

    def ready(self):
        # 여기서는 시그널 연결만 수행
        post_migrate.connect(self.on_post_migrate, sender=self)

    def on_post_migrate(self, sender, **kwargs):
        try:
            from .tasks import setup_bitcoin_analysis_task
            setup_bitcoin_analysis_task()
        except ProgrammingError:
            print("Warning: Django-Q 테이블이 존재하지 않습니다. 마이그레이션을 실행해주세요.")
        except ImportError as e:
            print(f"Error importing tasks module: {str(e)}")
        except Exception as e:
            print(f"Error: 스케줄 설정 중 오류 발생: {str(e)}")