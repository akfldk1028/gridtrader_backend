from django.apps import AppConfig
from django.db.utils import OperationalError, ProgrammingError
from django.db.models.signals import post_migrate


class BinanaceaccountConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'binanaceAccount'

    def ready(self):
        # 여기서는 시그널 연결만 수행
        post_migrate.connect(self.on_post_migrate, sender=self)

    def on_post_migrate(self, sender, **kwargs):
        # 데이터베이스 연결 확인
        # Django-Q 스케줄 설정

        try:
            from .tasks import setup_update_account_info_task
            setup_update_account_info_task()
        except ProgrammingError:
            print("Warning: Django-Q 테이블이 존재하지 않습니다. 마이그레이션을 실행해주세요.")
        except ImportError:
            print("Error: tasks 모듈을 가져올 수 없습니다.")
        except Exception as e:
            print(f"Error: 스케줄 설정 중 오류 발생: {str(e)}")