from django.db import models
from django.contrib.auth.models import User
from common.models import CommonModel

class BinanceFuturePosition(models.Model):
    symbol = models.CharField(max_length=20, help_text="거래 쌍 심볼")
    position_amt = models.DecimalField(max_digits=30, decimal_places=8, help_text="포지션 수량")
    entry_price = models.DecimalField(max_digits=30, decimal_places=8, help_text="진입 가격")
    mark_price = models.DecimalField(max_digits=30, decimal_places=8, help_text="마크 가격")
    un_realized_profit = models.DecimalField(max_digits=30, decimal_places=8, help_text="미실현 손익")
    leverage = models.IntegerField(help_text="레버리지")
    profit_percentage = models.DecimalField(max_digits=30, decimal_places=8, help_text="수익률 (%)")


class SpotBalance(models.Model):
    asset = models.CharField(max_length=10, help_text="자산 심볼")
    free = models.DecimalField(max_digits=30, decimal_places=8, help_text="사용 가능한 자산 수량")
    locked = models.DecimalField(max_digits=30, decimal_places=8, help_text="잠긴 자산 수량")

class FutureBalance(models.Model):
    positions = models.ManyToManyField(BinanceFuturePosition, related_name='future_positions', help_text="연관된 선물 포지션들", blank=True)
    asset = models.CharField(max_length=10, help_text="자산 심볼 (USDT)")
    balance = models.DecimalField(max_digits=30, decimal_places=8, help_text="전체 잔액")
    cross_wallet_balance = models.DecimalField(max_digits=30, decimal_places=8, help_text="교차 마진 지갑 잔액")
    cross_un_pnl = models.DecimalField(max_digits=30, decimal_places=8, help_text="교차 마진 미실현 손익")
    available_balance = models.DecimalField(max_digits=30, decimal_places=8, help_text="사용 가능한 잔액")
    max_withdraw_amount = models.DecimalField(max_digits=30, decimal_places=8, help_text="최대 출금 가능 금액")
    margin_available = models.BooleanField(help_text="마진 사용 가능 여부")
    update_time = models.BigIntegerField(null=True, blank=True, help_text="마지막 업데이트 시간 (밀리초)")


class BinanceAccount(models.Model):
    ACCOUNT_TYPES = [
        ('SPOT', 'Spot'),
        ('FUTURES', 'Futures'),
    ]
    spotbalance = models.ForeignKey(SpotBalance, on_delete=models.CASCADE, null=True, blank=True, related_name='spot_balances', help_text="연관된 현물 계정")
    account_type = models.CharField(max_length=10, choices=ACCOUNT_TYPES, default='SPOT', help_text="계정 유형 (현물 또는 선물)")
    can_trade = models.BooleanField(help_text="거래 가능 여부")
    can_withdraw = models.BooleanField(help_text="출금 가능 여부")
    can_deposit = models.BooleanField(help_text="입금 가능 여부")
    update_time = models.BigIntegerField(help_text="마지막 업데이트 시간 (밀리초)")
    maker_commission = models.IntegerField(help_text="메이커 수수료 비율")
    taker_commission = models.IntegerField(help_text="테이커 수수료 비율")

class BinanceFuturesAccount(models.Model):
    futurebalance = models.ForeignKey(FutureBalance, on_delete=models.CASCADE,null=True, blank=True, related_name='future_balance', help_text="연관된 현물 계정")
    account_type = models.CharField(max_length=10, default='FUTURES', help_text="계정 유형 (항상 'FUTURES')")
    total_wallet_balance = models.DecimalField(max_digits=30, decimal_places=8, help_text="전체 지갑 잔액")
    total_unrealized_profit = models.DecimalField(max_digits=30, decimal_places=8, help_text="전체 미실현 손익")
    total_margin_balance = models.DecimalField(max_digits=30, decimal_places=8, help_text="전체 마진 잔액")
    available_balance = models.DecimalField(max_digits=30, decimal_places=8, help_text="사용 가능한 잔액")
    max_withdraw_amount = models.DecimalField(max_digits=30, decimal_places=8, help_text="최대 출금 가능 금액")




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
