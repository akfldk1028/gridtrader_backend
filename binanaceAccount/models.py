from django.db import models
from django.contrib.auth.models import User
from common.models import CommonModel
from django.utils import timezone







class DailyBalance(CommonModel):
    futures_balance = models.JSONField()
    futures_positions = models.JSONField()

    def __str__(self):
        return f"Balance for {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"

    class Meta:
        ordering = ['-created_at']
        get_latest_by = 'created_at'




class BinanceOrder(CommonModel):
    symbol = models.CharField(max_length=20, help_text="거래 쌍 심볼")
    order_id = models.BigIntegerField(help_text="Binance에서 제공한 고유 주문 ID")
    side = models.CharField(max_length=50, help_text="주문 방향 (매수 또는 매도)")
    type = models.CharField(max_length=20, help_text="주문 유형 (예: 시장가, 지정가)")
    quantity = models.DecimalField(max_digits=30, decimal_places=8, help_text="주문 수량")
    reduce_only = models.BooleanField(help_text="포지션 감소 전용 주문 여부")
    price = models.DecimalField(max_digits=30, decimal_places=8, null=True, blank=True, help_text="주문 가격 (해당되는 경우)")
    status = models.CharField(max_length=20, help_text="주문의 현재 상태")

class BinanceSymbolSettings(CommonModel):
    symbol = models.CharField(max_length=20, help_text="거래 쌍 심볼")
    leverage = models.IntegerField(help_text="이 심볼에 대한 현재 레버리지 설정")
    margin_type = models.CharField(max_length=30, help_text="마진 유형 (격리 또는 교차)")
