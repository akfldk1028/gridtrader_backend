from django.contrib import admin
from .models import TradingRecord

@admin.register(TradingRecord)
class TradingRecordAdmin(admin.ModelAdmin):
    list_display = ('id', 'symbols', 'created_at')
    search_fields = ('symbols',)
