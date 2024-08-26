# serializers.py
from rest_framework import serializers

from .models import BinanceAccount, BinanceFuturesAccount, SpotBalance, FutureBalance, BinanceFuturePosition, BinanceOrder, BinanceSymbolSettings


class SpotBalanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = SpotBalance
        fields = ['asset', 'free', 'locked']

class BinanceAccountSerializer(serializers.ModelSerializer):
    spotbalance = SpotBalanceSerializer()

    class Meta:
        model = BinanceAccount
        fields = ['account_type', 'can_trade', 'can_withdraw', 'can_deposit', 'update_time', 'maker_commission', 'taker_commission', 'spotbalance']

class BinanceFuturePositionSerializer(serializers.ModelSerializer):
    class Meta:
        model = BinanceFuturePosition
        fields = ['symbol', 'position_amt', 'entry_price', 'mark_price', 'un_realized_profit', 'leverage', 'profit_percentage']

class FutureBalanceSerializer(serializers.ModelSerializer):
    futurePosition = BinanceFuturePositionSerializer()

    class Meta:
        model = FutureBalance
        fields = ['asset', 'balance', 'cross_wallet_balance', 'cross_un_pnl', 'available_balance', 'max_withdraw_amount', 'margin_available', 'update_time', 'futurePosition']

class BinanceFuturesAccountSerializer(serializers.ModelSerializer):
    futurebalance = FutureBalanceSerializer()

    class Meta:
        model = BinanceFuturesAccount
        fields = ['account_type', 'total_wallet_balance', 'total_unrealized_profit', 'total_margin_balance', 'available_balance', 'max_withdraw_amount', 'futurebalance']


class BinanceOrderSerializer(serializers.ModelSerializer):

    class Meta:
        model = BinanceOrder
        fields = '__all__'

class BinanceSymbolSettingsSerializer(serializers.ModelSerializer):

    class Meta:
        model = BinanceSymbolSettings
        fields = '__all__'
