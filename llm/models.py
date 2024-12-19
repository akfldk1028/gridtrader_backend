from django.db import models
from common.models import CommonModel
from decimal import Decimal


class AnalysisResult(CommonModel):
    date = models.DateTimeField(auto_now_add=True )
    symbol = models.CharField(max_length=20 , null=True, blank=True)
    korean_summary = models.TextField(null=True, blank=True)
    result_string = models.TextField(null=True, blank=True)
    current_price = models.FloatField(default=0, null=True, blank=True)
    price_prediction = models.CharField(max_length=10, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    selected_strategy = models.CharField(max_length=20)
    analysis_results_30m = models.TextField(null=True, blank=True)
    analysis_results_1hour = models.TextField(null=True, blank=True)
    analysis_results_daily = models.TextField(null=True, blank=True)


    def __str__(self):
        return f"{self.symbol} - {self.date}"

class Analysis(CommonModel):
    TRADE_TYPES = [
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
        ('HOLD', 'Hold')
    ]


    # 분석 관련 필드
    korean_summary = models.TextField(null=True, blank=True)
    result_string = models.TextField(null=True, blank=True)
    current_price = models.FloatField(default=0, null=True, blank=True)
    price_prediction = models.CharField(max_length=10, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    selected_strategy = models.CharField(max_length=20)

    # 기본 필드
    symbol = models.CharField(max_length=20 , null=True, blank=True)
    trade_type = models.CharField('거래 유형', max_length=4, choices=TRADE_TYPES, default='HOLD')
    trade_amount_krw = models.DecimalField('거래 금액', max_digits=20, decimal_places=2, default=Decimal('0.00'))
    coin_balance = models.DecimalField('코인 보유량', max_digits=20, decimal_places=8, default=Decimal('0.00000000'))

    # 잔고 관련 필드
    coin_balance = models.DecimalField('코인 보유량', max_digits=20, decimal_places=8, default=Decimal('0.00000000'))
    balance = models.DecimalField('보유 자산', max_digits=20, decimal_places=2, default=Decimal('0.00'))
    current_price = models.DecimalField('현재 가격', max_digits=20, decimal_places=2, default=Decimal('0.00'))
    avg_buy_price = models.DecimalField('평균 매수가', max_digits=20, decimal_places=2, default=Decimal('0.00'))


    def __str__(self):
        return f"{self.symbol}"

