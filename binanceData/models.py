from django.db import models
from common.models import CommonModel

# Create your models here.

class TradingRecord(CommonModel):
    symbols = models.JSONField(null=True, blank=True)  # nullable 설정

    def __str__(self):
        return f"{self.symbols}"