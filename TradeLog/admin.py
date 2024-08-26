from django.contrib import admin
from .models import Log, Order, Strategy

@admin.register(Log)
class LogAdmin(admin.ModelAdmin):
    list_display = ('time','gateway_name', 'msg')
    list_filter = ('gateway_name',)
    search_fields = ('msg',)

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'type', 'direction', 'status', 'price', 'volume', 'traded')
    list_filter = ('status', 'type', 'direction')
    search_fields = ('symbol',)

@admin.register(Strategy)
class StrategyAdmin(admin.ModelAdmin):
    list_display = ('strategy_name', 'vt_symbol', 'class_name', 'author')
    search_fields = ('strategy_name', 'class_name')