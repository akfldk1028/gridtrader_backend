from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/data/(?P<token>\w+)/$', consumers.DataConsumer.as_asgi()),
]