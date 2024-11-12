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
    trade_type = models.CharField('거래 유형', max_length=4, choices=TRADE_TYPES)
    trade_ratio = models.DecimalField('거래 비율(%)', max_digits=5, decimal_places=4)
    trade_amount_krw = models.DecimalField('거래 금액(KRW)', max_digits=20, decimal_places=2)
    trade_reason = models.TextField('거래 이유')
    coin_balance = models.DecimalField('코인 보유량', max_digits=20, decimal_places=8)
    balance = models.DecimalField('USDT/KRW 보유량', max_digits=20, decimal_places=2)
    current_price = models.DecimalField('현재 가격', max_digits=20, decimal_places=2)
    trade_reflection = models.TextField('거래 반성')


    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['-timestamp']),
            models.Index(fields=['coin_symbol']),
        ]