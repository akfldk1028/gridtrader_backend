from django.urls import path
# from .views import PhotoDetail, GetUploadURL
from . import views

urlpatterns = [
    path('<str:symbol>/<str:interval>/', views.BinanceChartDataAPIView.as_view()),
    path('llm-bitcoin-data/',  views.BinanceLLMChartDataAPIView.as_view(),),
    path('trendLines/<str:symbol>/<str:interval>/', views.TrendLinesAPIView.as_view(), ),

]