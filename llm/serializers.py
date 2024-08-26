from rest_framework import serializers
from .models import AnalysisResult

from rest_framework import serializers
from .models import AnalysisResult

class AnalysisResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisResult
        fields = ['date', 'symbol', 'result_string', 'current_price', 'price_prediction', 'confidence', 'selected_strategy']