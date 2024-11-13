from django.urls import path
# from .views import PhotoDetail, GetUploadURL
from . import views

urlpatterns = [
    path('positions', views.PositionsView.as_view(), name='positions'),
    path('leverage', views.LeverageView.as_view(), name='leverage'),
    path('open-order', views.OpenOrderView.as_view(), name='openorder'),
    path('close-order', views.CloseOrderView.as_view(), name='closeorder'),
    path('get-spot-info', views.SpotAccountInfoView.as_view(),),
    path('get-future-info', views.FuturesAccountInfoView.as_view(), ),
    path('get-spot-balance', views.SpotBalanceView.as_view(), ),
    path('get-future-balance', views.FuturesBalanceView.as_view(), ),
    path('get-future-position', views.FuturesPositionView.as_view(), ),
    path('get-server-time', views.ServerTimeView.as_view(), ),
    path('daily-balance', views.DailyBalanceView.as_view(), ),
    path('upbit/balance/', views.BalanceView.as_view(), name='balance'),
    path('upbit/trade/', views.TradeView.as_view(), name='trade'),
    path('upbit/price/', views.CurrentPriceView.as_view(), name='current-price'),
]