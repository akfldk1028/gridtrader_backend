from django.db import models
from common.models import CommonModel

# Create your models here.

class TradingRecord(CommonModel):
    symbols = models.JSONField(null=True, blank=True)  # nullable 설정

    def __str__(self):
        return f"{self.symbols}"


class BinanceTradingSummary(CommonModel):
    long_symbols = models.JSONField(default=list, blank=True)
    short_symbols = models.JSONField(default=list, blank=True)

    def __str__(self):
        return f"Trading Summary at Long {self.long_symbols} Short  {self.short_symbols}"