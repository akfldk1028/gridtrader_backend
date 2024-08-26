from django.db import models
from common.models import CommonModel


class AnalysisResult(CommonModel):
    date = models.DateTimeField(auto_now_add=True)
    symbol = models.CharField(max_length=20)
    result_string = models.TextField()
    current_price = models.FloatField()
    price_prediction = models.CharField(max_length=10)
    confidence = models.FloatField()
    selected_strategy = models.CharField(max_length=20)

    def __str__(self):
        return f"{self.symbol} - {self.date}"

