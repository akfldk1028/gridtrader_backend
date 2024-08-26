from django.urls import path
# from .views import PhotoDetail, GetUploadURL
from . import views

urlpatterns = [
    path('logs', views.LogDataView.as_view(), name='log-data'),
    path('orders', views.OrderDataView.as_view(), name='order-data'),
    path('strategies', views.StrategyDataView.as_view(), name='strategy-data'),
]