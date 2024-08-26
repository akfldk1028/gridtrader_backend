from django.urls import path
# from .views import PhotoDetail, GetUploadURL
from . import views

urlpatterns = [
    path('<str:symbol>/<str:interval>/', views.BinanceChartDataAPIView.as_view()),
]