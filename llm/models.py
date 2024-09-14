from django.db import models
from common.models import CommonModel


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

