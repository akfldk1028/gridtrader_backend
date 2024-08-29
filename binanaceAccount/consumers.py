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
import pandas as pd
from django.db import IntegrityError



class BinanceBaseConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.client = await AsyncClient.create(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
        await self.sync_server_time()

    async def disconnect(self, close_code):
        if hasattr(self, 'client'):
            await self.client.close_connection()

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

class PeriodicDataConsumer(BinanceBaseConsumer):
    async def connect(self):
        await super().connect()
        self.periodic_task = asyncio.create_task(self.periodically_send_data())

    async def disconnect(self, close_code):
        if hasattr(self, 'periodic_task'):
            self.periodic_task.cancel()
        await super().disconnect(close_code)

    async def periodically_send_data(self):
        while True:
            try:
                await self.get_futures_balance()
                await self.get_futures_positions('')
                await asyncio.sleep(2)  # Send data every 2 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in periodic task: {str(e)}")
                await asyncio.sleep(5)  # Wait 5 seconds before retrying in case of error



class OnDemandDataConsumer(BinanceBaseConsumer):
    async def connect(self):
        await super().connect()
        self.group_name = "binanceQ"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        print(f"OnDemandDataConsumer connected with channel name: {self.channel_name}")


    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)
        print(f"OnDemandDataConsumer disconnected with code: {close_code}")
        if hasattr(self, 'client'):
            await self.client.close_connection()
        await super().disconnect(close_code)


    async def save_daily_balance(self, event):
        print(f"save_daily_balance method called with event: {event}")
        try:
            # 여기에 기존 로직을 유지합니다
            await self.save_daily_balance_logic()
            print("save_daily_balance completed successfully")
        except Exception as e:
            print(f"Error in save_daily_balance: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error saving daily balance: {str(e)}"
            }))

    async def receive(self, text_data):
        data = json.loads(text_data)
        action = data.get('action')
        print(data)
        print(action)
        print("아아아시발 왜 안대")
        if action == 'get_bitcoin_data_and_price':
            symbol = data.get('symbol')
            await self.get_bitcoin_data_and_price(symbol)
        elif action == 'save_daily_balance':
            await self.save_daily_balance()


    async def get_bitcoin_data_and_price(self, symbol):
        try:
            hourly_candles = await self.client.get_klines(symbol=symbol, interval=self.client.KLINE_INTERVAL_1HOUR,
                                                          limit=500)
            daily_candles = await self.client.get_klines(symbol=symbol, interval=self.client.KLINE_INTERVAL_1DAY,
                                                         limit=500)
            ticker = await self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            def process_candles(candles):
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                    'quote_asset_volume', 'number_of_trades',
                                                    'taker_buy_base_asset_volume',
                                                    'taker_buy_quote_asset_volume', 'ignore'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(
                    float)

                # RSI 계산
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))

                # 스토캐스틱 계산
                low_14 = df['low'].rolling(window=14).min()
                high_14 = df['high'].rolling(window=14).max()
                df['%K'] = (df['close'] - low_14) / (high_14 - low_14) * 100
                df['%D'] = df['%K'].rolling(window=3).mean()

                return df.to_dict(orient='records')

            bitcoin_data = {
                'hourly': process_candles(hourly_candles),
                'daily': process_candles(daily_candles),
                'current_price': current_price
            }

            await self.send(text_data=json.dumps({
                'type': 'bitcoin_data_and_price',  # 'llm_data_and_price'에서 변경
                'data': bitcoin_data
            }))

        except BinanceAPIException as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error fetching Bitcoin data: {str(e)}"
            }))

    async def save_daily_balance_logic(self):
        # 기존의 save_daily_balance 메서드의 내용을 여기로 옮깁니다
        try:
            print("Starting save_daily_balance process")

            futures_balance = await self.get_futures_balance()
            print(f"Futures balance data: {futures_balance}")

            futures_positions = await self.get_futures_positions('')
            print(f"Futures positions data: {futures_positions}")

            DailyBalance = apps.get_model('binanaceAccount', 'DailyBalance')

            new_balance = await sync_to_async(DailyBalance.objects.create)(
                futures_balance=futures_balance,
                futures_positions=futures_positions
            )

            print(
                f"Successfully created DailyBalance record with id {new_balance.id} at {new_balance.created_at}")

            await self.send(text_data=json.dumps({
                'type': 'daily_balance_saved',
                'message': 'Daily balance saved successfully',
                'record_id': new_balance.id,
                'timestamp': new_balance.created_at.isoformat()
            }))
        except Exception as e:
            print(f"Unexpected error in save_daily_balance: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Unexpected error saving daily balance. Please check the logs."
            }))

