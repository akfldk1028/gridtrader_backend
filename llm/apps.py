from django.apps import AppConfig
from django.db.models.signals import post_migrate


class LlmConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'llm'

    def ready(self):
        # 여기서는 시그널 연결만 수행
        post_migrate.connect(self.on_post_migrate, sender=self)

    def on_post_migrate(self, sender, **kwargs):
        from .tasks import setup_bitcoin_analysis_task
        setup_bitcoin_analysis_task()