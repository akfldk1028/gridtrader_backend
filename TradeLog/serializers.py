from rest_framework import serializers
from .models import Log, Order, Strategy


class LogSerializer(serializers.Serializer):
    msg = serializers.CharField(max_length=1000)
    gateway_name = serializers.CharField(max_length=100)
    time = serializers.DateTimeField()

class OrderSerializer(serializers.Serializer):
    symbol = serializers.CharField(max_length=100)
    orderid = serializers.CharField(max_length=100)
    type = serializers.CharField(max_length=50)
    direction = serializers.CharField(max_length=50)
    offset = serializers.CharField(max_length=50, allow_blank=True)
    price = serializers.FloatField()
    volume = serializers.FloatField()
    traded = serializers.FloatField()
    status = serializers.CharField(max_length=50)
    gateway_name = serializers.CharField(max_length=100)

class StrategySerializer(serializers.Serializer):
    strategy_name = serializers.CharField(max_length=100)
    # 필요한 다른 필드들 추가

# class LogSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Log
#         fields = '__all__'
#
# class OrderSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Order
#         fields = '__all__'
#
# class StrategySerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Strategy
#         fields = '__all__'