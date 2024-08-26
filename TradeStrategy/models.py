from django.db import models

class StrategyConfig(models.Model):
    name = models.CharField(max_length=100, unique=True)
    config = models.JSONField()  # 이 부분이 변경되었습니다

    def __str__(self):
        return self.name