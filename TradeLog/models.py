from django.db import models
class Log(models.Model):
    msg = models.TextField()
    gateway_name = models.CharField(max_length=100)
    time = models.DateTimeField()

    def __str__(self):
        return f"{self.gateway_name}"

class Order(models.Model):
    orderid = models.CharField(max_length=100, unique=True)
    symbol = models.CharField(max_length=100)
    type = models.CharField(max_length=50)
    direction = models.CharField(max_length=50)
    offset = models.CharField(max_length=50, null=True, blank=True)
    price = models.FloatField()
    volume = models.FloatField()
    traded = models.FloatField()
    status = models.CharField(max_length=50)
    gateway_name = models.CharField(max_length=100)



    def __str__(self):
        return f"{self.symbol} - {self.status}"

class Strategy(models.Model):
    strategy_name = models.CharField(max_length=100, unique=True)
    vt_symbol = models.CharField(max_length=100)
    class_name = models.CharField(max_length=100)
    author = models.CharField(max_length=100)
    parameters = models.JSONField()
    variables = models.JSONField()

    def __str__(self):
        return self.strategy_name