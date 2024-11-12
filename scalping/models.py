# scalping/models.py
from django.db import models
from decimal import Decimal
from common.models import CommonModel


class TradingRecord(CommonModel):
    TRADE_TYPES = [
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
        ('HOLD', 'Hold')
    ]

    timestamp = models.DateTimeField('거래 시간')
    coin_symbol = models.CharField('코인 심볼', max_length=20)
    trade_type = models.CharField('거래 유형', max_length=4, choices=TRADE_TYPES, default='HOLD')
    trade_ratio = models.DecimalField('거래 비율(%)', max_digits=5, decimal_places=4, default=Decimal('0.0000'))
    trade_amount_krw = models.DecimalField('거래 금액(KRW)', max_digits=20, decimal_places=2, default=Decimal('0.00'))
    trade_reason = models.TextField('거래 이유', default='')
    coin_balance = models.DecimalField('코인 보유량', max_digits=20, decimal_places=8, default=Decimal('0.00000000'))
    balance = models.DecimalField('USDT/KRW 보유량', max_digits=20, decimal_places=2, default=Decimal('0.00'))
    current_price = models.DecimalField('현재 가격', max_digits=20, decimal_places=2, default=Decimal('0.00'))
    trade_reflection = models.TextField('거래 반성', default='')

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['-timestamp']),
            models.Index(fields=['coin_symbol']),
        ]


class CoinScalpingAnalysis(CommonModel):
    RECOMMENDATION_LEVELS = [
        ('HIGH', 'High Priority'),
        ('MEDIUM', 'Medium Priority'),
        ('LOW', 'Low Priority')
    ]

    timestamp = models.DateTimeField('분석 시간', auto_now_add=True)
    coin_symbol = models.CharField('코인 심볼', max_length=20)
    current_price = models.DecimalField('현재 가격', max_digits=20, decimal_places=2, default=Decimal('0.00'))
    volume_24h = models.DecimalField('24시간 거래량', max_digits=20, decimal_places=2, default=Decimal('0.00'))
    price_change_24h = models.DecimalField('24시간 변동률(%)', max_digits=5, decimal_places=2, default=Decimal('0.00'))
    scalping_score = models.DecimalField('스캘핑 점수', max_digits=5, decimal_places=2, default=Decimal('0.00'))
    priority = models.CharField('우선순위', max_length=10, choices=RECOMMENDATION_LEVELS)
    analysis = models.TextField('분석 결과')

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['coin_symbol']),
            models.Index(fields=['priority']),
        ]