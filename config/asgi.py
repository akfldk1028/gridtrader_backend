"""
ASGI config for config project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

# import os
# from django.core.asgi import get_asgi_application
# from channels.routing import ProtocolTypeRouter, URLRouter
# from channels.auth import AuthMiddlewareStack
# from binanaceAccount.routing import websocket_urlpatterns
#
#
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
# application = ProtocolTypeRouter({
#     "http": get_asgi_application(),
#     "websocket": AuthMiddlewareStack(
#         URLRouter(
#             websocket_urlpatterns
#         )
#     ),
# })

import os
import django
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from binanaceAccount.routing import websocket_urlpatterns as binance_websocket_urlpatterns
from scalping.routing import websocket_urlpatterns as scalping_websocket_urlpatterns

# 두 앱의 URL 패턴 결합
combined_patterns = binance_websocket_urlpatterns + scalping_websocket_urlpatterns

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(combined_patterns)
    ),
})