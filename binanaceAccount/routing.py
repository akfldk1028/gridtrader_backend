from django.urls import re_path
from .consumers import BinanceAPIConsumer

websocket_urlpatterns = [
    re_path(r'ws/binance/$', BinanceAPIConsumer.as_asgi()),
]