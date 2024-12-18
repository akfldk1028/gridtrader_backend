from django.contrib import admin
from .models import TradingRecord
from .models import BinanceTradingSummary, KoreaStockData, StockData

@admin.register(TradingRecord)
class TradingRecordAdmin(admin.ModelAdmin):
    list_display = ('id', 'symbols', 'created_at')
    search_fields = ('symbols',)
@admin.register(KoreaStockData)
class KoreaStockDataAdmin(admin.ModelAdmin):
    list_display = ('id', 'symbols', 'created_at')
    search_fields = ('symbols',)

@admin.register(StockData)
class StockDataAdmin(admin.ModelAdmin):
    list_display = ('id', 'symbols', 'created_at')
    search_fields = ('symbols',)

@admin.register(BinanceTradingSummary)
class BinanceTradingSummaryAdmin(admin.ModelAdmin):
    list_display = ('id', 'created_at', 'get_long_symbols', 'get_short_symbols')
    search_fields = ('long_symbols', 'short_symbols')
    list_filter = ('created_at',)

    def get_long_symbols(self, obj):
        return ", ".join(obj.long_symbols)
    get_long_symbols.short_description = 'Long Symbols'

    def get_short_symbols(self, obj):
        return ", ".join(obj.short_symbols)
    get_short_symbols.short_description = 'Short Symbols'