from django.urls import re_path
from .consumers import BinanceDataConsumer
websocket_urlpatterns = [
    re_path(r'ws/binance/$', BinanceDataConsumer.as_asgi()),
    # re_path(r'ws/markprice/$', MarkPriceConsumer.as_asgi()),

    # re_path(r'ws/account/$', OnDemandDataConsumer.as_asgi()),
]