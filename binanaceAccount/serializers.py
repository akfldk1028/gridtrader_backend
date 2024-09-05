# serializers.py
from rest_framework import serializers

from .models import BinanceOrder, BinanceSymbolSettings, DailyBalance



class BinanceOrderSerializer(serializers.ModelSerializer):

    class Meta:
        model = BinanceOrder
        fields = '__all__'

class BinanceSymbolSettingsSerializer(serializers.ModelSerializer):

    class Meta:
        model = BinanceSymbolSettings
        fields = '__all__'


class DailyBalanceSerializer(serializers.ModelSerializer):

    class Meta:
        model = DailyBalance
        fields = '__all__'
