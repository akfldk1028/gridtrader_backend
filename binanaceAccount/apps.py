from django.apps import AppConfig


class BinanaceaccountConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'binanaceAccount'

    def ready(self):
        from .tasks import setup_update_account_info_task
        setup_update_account_info_task()