from django.contrib import admin
from .models import AnalysisResult,Analysis

@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'date', 'current_price', 'price_prediction', 'confidence', 'selected_strategy')
    list_filter = ('symbol', 'price_prediction', 'selected_strategy')
    search_fields = ('symbol',)


@admin.register(Analysis)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'current_price','result_string', 'selected_strategy', 'balance', 'coin_balance')
    list_filter = ('symbol',  'selected_strategy')
    search_fields = ('symbol',)