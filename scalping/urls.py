from django.urls import path
# from .views import PhotoDetail, GetUploadURL
from . import views
from .views import WebSocketTestView

urlpatterns = [
    path('ws/', WebSocketTestView.as_view(), name='websocket'),

]