from django.urls import re_path
from .consumers import PeriodicDataConsumer , OnDemandDataConsumer

websocket_urlpatterns = [
    re_path(r'ws/binance/$', PeriodicDataConsumer.as_asgi()),
    re_path(r'ws/binanceQ/', OnDemandDataConsumer.as_asgi()),
]