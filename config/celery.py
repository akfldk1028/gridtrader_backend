# from __future__ import absolute_import, unicode_literals
# import os
# import uuid
#
# from celery import Celery
# from celery.schedules import crontab
# from datetime import timedelta

# from binanaceAccount.tasks.update_account_info import update_account_info


# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
#
# # Celery 앱 생성
# app = Celery('config')
# app.config_from_object('django.conf:settings', namespace='CELERY')
# app.autodiscover_tasks()
# # app = Celery()
#
#
# @app.on_after_configure.connect
# def setup_periodic_tasks(sender, **kwargs):
#     # Calls test('hello') every 10 seconds.
#     sender.add_periodic_task(10.0, test.s('hello'), name='add every 10')
#
#     # Calls test('hello') every 30 seconds.
#     # It uses the same signature of previous task, an explicit name is
#     # defined to avoid this task replacing the previous one defined.
#     sender.add_periodic_task(30.0, test.s('hello'), name='add every 30')
#
#     # Calls test('world') every 30 seconds
#     sender.add_periodic_task(30.0, test.s('world'), expires=10)
#
#
#
#
# @app.task
# def test(arg):
#     print(arg)
#
#
# @app.task
# def add(x, y):
#     z = x + y
#     print(z)
#
# app.conf.beat_schedule = {
#     # Executes every Monday morning at 7:30 a.m.
#     'add-every-monday-morning': {
#         'task': 'tasks.add',
#         'schedule': crontab(minute='*/2'),
#         'args': (16, 16),
#     },
# }
# app.conf.beat_schedule = {
#     'update-account-info-every-two-minutes': {
#         'task': 'binanaceAccount.tasks.update_account_info',
#         'schedule': crontab(minute='*/2'),
#     },
# }
