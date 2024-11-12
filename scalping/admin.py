# scalping/admin.py
from django.contrib import admin
from django.utils.html import format_html
from .models import TradingRecord


@admin.register(TradingRecord)
class TradingRecordAdmin(admin.ModelAdmin):
    list_display = [
        'timestamp',
        'coin_symbol',
        'trade_type_colored',
        'trade_ratio_formatted',
        'trade_amount_formatted',
        'current_price_formatted',
        'coin_balance_formatted',
        'balance_formatted'
    ]

    list_filter = [
        'trade_type',
        'coin_symbol',
        ('timestamp', admin.DateFieldListFilter),
    ]

    search_fields = [
        'coin_symbol',
        'trade_reason',
        'trade_reflection'
    ]

    readonly_fields = ['timestamp']

    fieldsets = (
        ('거래 기본 정보', {
            'fields': (
                'timestamp',
                'coin_symbol',
                'trade_type',
                'current_price',
            )
        }),
        ('거래 상세', {
            'fields': (
                'trade_ratio',
                'trade_amount_krw',
                'trade_reason',
            )
        }),
        ('계좌 상태', {
            'fields': (
                'coin_balance',
                'balance',
            )
        }),
        ('분석', {
            'fields': (
                'trade_reflection',
            ),
            'classes': ('collapse',)
        }),
    )

    def trade_type_colored(self, obj):
        colors = {
            'BUY': 'green',
            'SELL': 'red',
            'HOLD': 'gray'
        }
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            colors.get(obj.trade_type, 'black'),
            obj.get_trade_type_display()
        )

    trade_type_colored.short_description = '거래 유형'

    def trade_ratio_formatted(self, obj):
        return f"{obj.trade_ratio}%"

    trade_ratio_formatted.short_description = '거래 비율'

    def trade_amount_formatted(self, obj):
        return f"{int(obj.trade_amount_krw):,} KRW"

    trade_amount_formatted.short_description = '거래 금액'

    def current_price_formatted(self, obj):
        return f"{int(obj.current_price):,} KRW"

    current_price_formatted.short_description = '현재 가격'

    def coin_balance_formatted(self, obj):
        return f"{obj.coin_balance:.8f}"

    coin_balance_formatted.short_description = '코인 보유량'

    def balance_formatted(self, obj):
        return f"{int(obj.balance):,} KRW"

    balance_formatted.short_description = 'KRW 잔고'

