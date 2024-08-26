from django.urls import path
from .views import StrategyConfigView

urlpatterns = [
    path('strategy', StrategyConfigView.as_view(), name='strategy_list'),
    path('strategy/<str:strategy_name>', StrategyConfigView.as_view(), name='strategy_detail'),
]