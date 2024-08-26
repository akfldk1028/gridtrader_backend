from django.contrib import admin
from .models import AnalysisResult

@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'date', 'current_price', 'price_prediction', 'confidence', 'selected_strategy')
    list_filter = ('symbol', 'price_prediction', 'selected_strategy')
    search_fields = ('symbol',)
