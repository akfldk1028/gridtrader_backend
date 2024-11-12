from django.contrib import admin
from .models import BinanceOrder, BinanceSymbolSettings, DailyBalance



@admin.register(DailyBalance)
class DailyBalanceAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'get_futures_balance', 'get_futures_positions')
    list_filter = ('created_at',)
    search_fields = ('created_at',)
    ordering = ('-created_at',)

    def get_futures_balance(self, obj):
        return obj.futures_balance.get('balance', 'N/A')
    get_futures_balance.short_description = 'Futures Balance'

    def get_futures_positions(self, obj):
        positions = obj.futures_positions
        if isinstance(positions, list):
            return ', '.join([f"{pos.get('symbol', 'N/A')}: {pos.get('positionAmt', 'N/A')}" for pos in positions[:3]])
        return 'N/A'
    get_futures_positions.short_description = 'Futures Positions (Top 3)'

@admin.register(BinanceOrder)
class BinanceOrderAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'order_id', 'side', 'type', 'quantity', 'status')
    list_filter = ('symbol', 'side', 'type', 'status')
    search_fields = ('symbol', 'order_id')

@admin.register(BinanceSymbolSettings)
class BinanceSymbolSettingsAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'leverage', 'margin_type')
    list_filter = ('symbol', 'leverage', 'margin_type')
    search_fields = ('symbol',)



# class CustomPeriodicTaskAdmin(BasePeriodicTaskAdmin):
#     actions = ['setup_periodic_tasks']
#
#     def setup_periodic_tasks(self, request, queryset):
#         schedule, created = IntervalSchedule.objects.get_or_create(every=60, period=IntervalSchedule.SECONDS)
#
#         task_name = 'update_account_info'
#         if PeriodicTask.objects.filter(name=task_name).exists():
#             p_test = PeriodicTask.objects.get(name=task_name)
#             p_test.enabled = True
#             p_test.interval = schedule
#             p_test.save()
#         else:
#             PeriodicTask.objects.create(
#                 interval=schedule,
#                 name=task_name,
#                 task='binanaceAccount.tasks.update_account_info',
#             )
#
#         self.message_user(request, "Periodic tasks have been set up or updated.")
#         return HttpResponse("Periodic tasks have been set up or updated.")
#
#     setup_periodic_tasks.short_description = "Setup or update periodic tasks"
#
# admin.site.unregister(PeriodicTask)
# admin.site.register(PeriodicTask, CustomPeriodicTaskAdmin)