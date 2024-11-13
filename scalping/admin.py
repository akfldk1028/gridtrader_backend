# scalping/admin.py
from django.contrib import admin
from .models import TradingRecord,CoinScalpingAnalysis
from django.utils.html import format_html


@admin.register(TradingRecord)
class TradingRecordAdmin(admin.ModelAdmin):
    list_display = [
        'formatted_created_at',  # created_at을 포맷팅해서 보여줌
        'exchange',
        'coin_symbol',
        'trade_type',
        'trade_ratio',
        'truncated_trade_reason',  # trade_reason 추가
        'trade_amount_display',
        'current_price_display',
        'coin_balance',
        'balance_display'
    ]

    list_filter = [
        'exchange',
        'trade_type',
        'coin_symbol',
        ('created_at', admin.DateFieldListFilter)
    ]

    search_fields = [
        'coin_symbol',
        'trade_reason',
        'trade_reflection'
    ]

    readonly_fields = [
        'created_at',
        'updated_at',
        'technical_indicators'
    ]

    ordering = ['-created_at']

    fieldsets = (
        ('기본 정보', {
            'fields': (
                'exchange',
                'coin_symbol',
                'trade_type',
                'created_at',
                'updated_at'
            )
        }),
        ('거래 정보', {
            'fields': (
                'trade_ratio',
                'trade_amount_krw',
                'current_price',
                'avg_buy_price'
            )
        }),
        ('잔고 정보', {
            'fields': (
                'coin_balance',
                'balance'
            )
        }),
        ('분석 정보', {
            'fields': (
                'trade_reason',
                'trade_reflection',
                'technical_indicators'
            ),
            'classes': ('collapse',)
        })
    )
    def truncated_trade_reason(self, obj):
        """거래 이유를 일정 길이로 자르고 툴팁으로 전체 내용 표시"""
        if len(obj.trade_reason) > 500:
            return format_html(
                '<span title="{}">{}</span>',
                obj.trade_reason,
                f"{obj.trade_reason[:500]}..."
            )
        return obj.trade_reason
    truncated_trade_reason.short_description = '거래 이유'

    def trade_amount_display(self, obj):
        """거래 금액 표시 (천 단위 구분)"""
        return f"{obj.trade_amount_krw:,.2f}"
    trade_amount_display.short_description = '거래 금액'

    def current_price_display(self, obj):
        """현재 가격 표시 (천 단위 구분)"""
        return f"{obj.current_price:,.2f}"
    current_price_display.short_description = '현재 가격'

    def balance_display(self, obj):
        """보유 자산 표시 (천 단위 구분)"""
        return f"{obj.balance:,.2f}"
    balance_display.short_description = '보유 자산'

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