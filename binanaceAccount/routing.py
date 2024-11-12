from django.urls import re_path
from .consumers import BinanceWebSocketConsumer
websocket_urlpatterns = [
    re_path(r'ws/binance/$', BinanceWebSocketConsumer.as_asgi()),
    # re_path(r'ws/markprice/$', MarkPriceConsumer.as_asgi()),

    # re_path(r'ws/account/$', OnDemandDataConsumer.as_asgi()),
]