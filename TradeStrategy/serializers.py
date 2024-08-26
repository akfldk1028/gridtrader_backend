from rest_framework import serializers
from .models import StrategyConfig

class StrategyConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = StrategyConfig
        fields = ['name', 'config']