# scalping/admin.py
from django.contrib import admin
from .models import TradingRecord,CoinScalpingAnalysis

@admin.register(TradingRecord)
class TradingRecordAdmin(admin.ModelAdmin):
    # 목록에 표시할 필드
    list_display = [
        'timestamp',
        'coin_symbol',
        'trade_type',
        'trade_ratio',
        'trade_amount_krw',
        'current_price',
        'coin_balance',
        'balance'
    ]

    # 필터 옵션
    list_filter = ['trade_type', 'coin_symbol']

    # 검색 필드
    search_fields = ['coin_symbol', 'trade_reason']

    # 읽기 전용 필드
    readonly_fields = ['timestamp']

    # 시간 순 정렬
    ordering = ['-timestamp']


@admin.register(CoinScalpingAnalysis)
class CoinScalpingAnalysisAdmin(admin.ModelAdmin):
    list_display = [
        'timestamp',
        'coin_symbol',
        'current_price',
        'price_change_24h',
        'scalping_score',
        'priority'
    ]
    list_filter = ['priority', 'coin_symbol']
    search_fields = ['coin_symbol', 'analysis']
    readonly_fields = ['timestamp']