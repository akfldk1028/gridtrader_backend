import json
from channels.generic.websocket import AsyncWebsocketConsumer
from binance.client import AsyncClient
from binance.exceptions import BinanceAPIException
from decimal import Decimal
import asyncio
from django.conf import settings
import time
from decimal import Decimal, InvalidOperation
from asgiref.sync import sync_to_async
from .models import DailyBalance
from django.utils import timezone
from django.apps import apps





class BinanceAPIConsumer(AsyncWebsocketConsumer):


    async def connect(self):
        await self.accept()
        await self.channel_layer.group_add("binance_updates", self.channel_name)
        self.client = await AsyncClient.create(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
        self.tasks = set()
        await self.sync_server_time()
        self.periodic_task = asyncio.create_task(self.periodically_send_data())

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("binance_updates", self.channel_name)
        for task in self.tasks:
            task.cancel()
        self.periodic_task.cancel()  # 주기적 작업 취소
        await self.client.close_connection()
    async def periodically_send_data(self):
        while True:
            await self.get_futures_balance()
            await self.get_futures_positions('')
            await asyncio.sleep(1)  # 2초마다 데이터 전송

    async def receive(self, text_data):
        data = json.loads(text_data)
        action = data.get('action')

        if action == 'get_futures_balance':
            await self.get_futures_balance()
        elif action == 'get_futures_positions':
            symbols = data.get('symbols', '')
            await self.get_futures_positions(symbols)

    # async def receive(self, text_data):
    #     data = json.loads(text_data)
    #     action = data.get('action')
    #
    #     if action == 'get_futures_balance':
    #         task = asyncio.create_task(self.get_futures_balance())
    #         self.tasks.add(task)
    #         task.add_done_callback(self.tasks.discard)
    #     elif action == 'get_futures_positions':
    #         symbols = data.get('symbols', '')
    #         task = asyncio.create_task(self.get_futures_positions(symbols))
    #         self.tasks.add(task)
    #         task.add_done_callback(self.tasks.discard)
        # Add more actions as needed

    async def sync_server_time(self):
        try:
            server_time = await self.client.get_server_time()
            self.client.timestamp_offset = server_time['serverTime'] - int(time.time() * 1000)
            print(f"Synced server time. Offset: {self.client.timestamp_offset}ms")
        except BinanceAPIException as e:
            print(f"Error syncing server time: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error syncing server time: {str(e)}"
            }))

    async def get_futures_balance(self):
        try:
            futures_balances = await self.client.futures_account_balance()
            futures_usdt_balance = next((item for item in futures_balances if item["asset"] == "USDT"), None)
            await self.send(text_data=json.dumps({
                'type': 'futures_balance',
                'data': futures_usdt_balance
            }))
        except BinanceAPIException as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    async def get_futures_positions(self, symbols):
        try:
            params = {}
            if symbols:
                params['symbol'] = symbols.strip().upper()

            all_positions = await self.client.futures_position_information(**params)

            filtered_positions = [
                pos for pos in all_positions
                if (not symbols or pos["symbol"] in symbols.split(',')) and float(pos["positionAmt"]) != 0
            ]

            for pos in filtered_positions:
                pos['profit_percentage'] = self.calculate_profit_percentage(pos)

            await self.send(text_data=json.dumps({
                'type': 'futures_positions',
                'data': filtered_positions
            }))
        except BinanceAPIException as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    def calculate_profit_percentage(self, position):
        try:
            position_amt = Decimal(position.get('positionAmt', '0'))
            if position_amt == Decimal('0'):
                return Decimal('0')

            entry_price = Decimal(position.get('entryPrice', '0'))
            mark_price = Decimal(position.get('markPrice', '0'))
            leverage = Decimal(position.get('leverage', '1'))

            if entry_price == Decimal('0'):
                return Decimal('0')

            if position_amt > Decimal('0'):  # Long position
                profit_percentage = ((mark_price - entry_price) / entry_price) * 100 * leverage
            else:  # Short position
                profit_percentage = ((entry_price - mark_price) / entry_price) * 100 * leverage

            return float(profit_percentage.quantize(Decimal('0.01')))
        except (InvalidOperation, ZeroDivisionError):
            return 0

    async def save_daily_balance(self, event=None):
        try:
            futures_balance = await self.get_futures_balance()
            futures_positions = await self.get_futures_positions('')

            DailyBalance = apps.get_model('binanaceAccount', 'DailyBalance')

            await sync_to_async(DailyBalance.objects.update_or_create)(
                date=timezone.now().date(),
                defaults={
                    'futures_balance': futures_balance,
                    'futures_positions': futures_positions
                }
            )

            await self.send(text_data=json.dumps({
                'type': 'daily_balance_saved',
                'message': 'Daily balance saved successfully'
            }))
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error saving daily balance: {str(e)}"
            }))