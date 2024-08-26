from django.contrib import admin
from .models import BinanceAccount, BinanceFuturesAccount, SpotBalance, FutureBalance, BinanceFuturePosition, BinanceOrder, BinanceSymbolSettings

@admin.register(BinanceFuturePosition)
class BinanceFuturePositionAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'position_amt', 'entry_price', 'mark_price', 'un_realized_profit', 'leverage', 'profit_percentage')
    search_fields = ('symbol',)

@admin.register(SpotBalance)
class SpotBalanceAdmin(admin.ModelAdmin):
    list_display = ('asset', 'free', 'locked')
    search_fields = ('asset',)

@admin.register(FutureBalance)
class FutureBalanceAdmin(admin.ModelAdmin):
    list_display = ('asset', 'balance', 'cross_wallet_balance', 'cross_un_pnl', 'available_balance', 'max_withdraw_amount', 'margin_available', 'update_time')
    search_fields = ('asset',)

@admin.register(BinanceAccount)
class BinanceAccountAdmin(admin.ModelAdmin):
    list_display = ('account_type', 'can_trade', 'can_withdraw', 'can_deposit', 'update_time', 'maker_commission', 'taker_commission')
    list_filter = ('account_type', 'can_trade', 'can_withdraw', 'can_deposit')
    raw_id_fields = ('spotbalance',)

@admin.register(BinanceFuturesAccount)
class BinanceFuturesAccountAdmin(admin.ModelAdmin):
    list_display = ('account_type', 'total_wallet_balance', 'total_unrealized_profit', 'total_margin_balance', 'available_balance', 'max_withdraw_amount')
    raw_id_fields = ('futurebalance',)


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