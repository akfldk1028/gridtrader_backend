import asyncio
from django.conf import settings
import time
from decimal import Decimal, InvalidOperation
from binance import AsyncClient, BinanceSocketManager, ThreadedWebsocketManager
from channels.generic.websocket import AsyncWebsocketConsumer
import json
from threading import Thread
import websocket
import json
import threading
import hmac
import hashlib
import requests
from binance.exceptions import BinanceAPIException


class BinanceWebSocketConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.client = await AsyncClient.create(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
        self.bm = BinanceSocketManager(self.client)
        self.twm = ThreadedWebsocketManager(api_key=settings.BINANCE_API_KEY, api_secret=settings.BINANCE_API_SECRET)

        self.user_socket = None
        self.mark_price_socket = None
        self.last_sent_data = {}
        self.mark_prices = {}
        self.listen_key = None

        await self.start_user_data_stream()
        await self.start_mark_price_socket()
        asyncio.create_task(self.keep_listen_key_alive())

    async def disconnect(self, close_code):
        if self.user_socket:
            await self.user_socket.__aexit__(None, None, None)
        if self.mark_price_socket:
            await self.mark_price_socket.__aexit__(None, None, None)
        if hasattr(self, 'client'):
            await self.client.close_connection()


    async def start_user_data_stream(self):
        self.listen_key = await self.client.futures_stream_get_listen_key()
        self.user_socket = self.bm.futures_user_socket()

        asyncio.create_task(self.user_socket_listener())


    async def start_mark_price_socket(self):
        self.mark_price_socket = self.twm.start_all_mark_price_socket(
            callback=self.handle_mark_price_update,
            fast=True
        )
        asyncio.create_task(self.mark_price_socket_listener())

    async def keep_listen_key_alive(self):
        while True:
            await asyncio.sleep(30 * 60)  # 30 minutes
            try:
                await self.client.futures_stream_keep_alive_listen_key(self.listen_key)
            except Exception as e:
                print(f"Error keeping listen key alive: {e}")
                await self.start_user_data_stream()

    async def user_socket_listener(self):
        async with self.user_socket as tscm:
            while True:
                try:
                    res = await tscm.recv()
                    if res:
                        event_type = res.get('e')
                        if event_type == 'ACCOUNT_UPDATE':
                            await self.handle_account_update(res)
                        elif event_type == 'ORDER_TRADE_UPDATE':
                            await self.handle_order_update(res)
                except Exception as e:
                    print(f"Error in user socket: {e}")
                    await asyncio.sleep(5)
                    await self.start_user_data_stream()
                    break

    async def mark_price_socket_listener(self):
        async with self.mark_price_socket as mps:
            while True:
                try:
                    res = await mps.recv()
                    if res:
                        await self.handle_mark_price_update(res)
                except Exception as e:
                    print(f"Error in mark price socket: {e}")
                    await asyncio.sleep(5)
                    await self.start_mark_price_socket()
                    break

    async def handle_account_update(self, data):
        balances = data['a']['B']
        positions = data['a']['P']

        usdt_balance = next((b for b in balances if b['a'] == 'USDT'), None)
        if usdt_balance:
            await self.send_if_changed('futures_balance', {
                'asset': 'USDT',
                'balance': usdt_balance['wb'],
                'crossWalletBalance': usdt_balance['cw'],
                'availableBalance': usdt_balance['ab']
            })

        active_positions = [position for position in positions if float(position['pa']) != 0]
        if active_positions:
            positions_data = []
            for position in active_positions:
                position_data = {
                    'symbol': position['s'],
                    'positionAmt': position['pa'],
                    'entryPrice': position['ep'],
                    'unrealizedProfit': position['up'],
                    'leverage': position['l'],
                    'markPrice': self.mark_prices.get(position['s'], position['mp'])
                }
                position_data['profit_percentage'] = self.calculate_profit_percentage(position_data)
                positions_data.append(position_data)

            await self.send_if_changed('futures_positions', positions_data)

    async def handle_order_update(self, data):
        # 주문 업데이트에 대한 처리
        pass

    async def handle_mark_price_update(self, data):
        for item in data:
            self.mark_prices[item['s']] = item['p']
        # Mark price 업데이트 후 포지션 정보 갱신
        await self.update_positions_with_new_mark_prices()

    async def update_positions_with_new_mark_prices(self):
        if 'futures_positions' in self.last_sent_data:
            positions = self.last_sent_data['futures_positions']
            updated = False
            for pos in positions:
                if pos['symbol'] in self.mark_prices:
                    pos['markPrice'] = self.mark_prices[pos['symbol']]
                    pos['profit_percentage'] = self.calculate_profit_percentage(pos)
                    updated = True
            if updated:
                await self.send_if_changed('futures_positions', positions)

    async def send_if_changed(self, data_type, data):
        key = data_type
        if self.last_sent_data.get(key) != data:
            self.last_sent_data[key] = data
            await self.send(text_data=json.dumps({
                'type': data_type,
                'data': data
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

            if position_amt > Decimal('0'):
                profit_percentage = ((mark_price - entry_price) / entry_price) * 100 * leverage
            else:
                profit_percentage = ((entry_price - mark_price) / entry_price) * 100 * leverage

            return float(profit_percentage.quantize(Decimal('0.01')))
        except (InvalidOperation, ZeroDivisionError):
            return 0



# class BinanceWebSocketConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         await self.accept()
#         self.client = await AsyncClient.create(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
#         self.bm = BinanceSocketManager(self.client)
#         self.twm = ThreadedWebsocketManager(api_key=settings.BINANCE_API_KEY, api_secret=settings.BINANCE_API_SECRET)
#
#         self.user_socket = None
#         self.mark_price_socket = None
#         self.reconnecting = False
#         self.last_sent_data = {}
#         self.mark_prices = {}
#         await self.sync_server_time()
#         await self.start_user_socket()
#         await self.start_twm_in_thread()
#         await self.initial_data_fetch()
#         self.periodic_update_task = asyncio.create_task(self.periodic_account_update())
#
#     async def disconnect(self, close_code):
#         if self.user_socket:
#             await self.user_socket.__aexit__(None, None, None)
#         if self.twm:
#             self.twm.stop()
#         if hasattr(self, 'client'):
#             await self.client.close_connection()
#         self.reconnecting = False
#
#         # Cancel periodic update task
#         if hasattr(self, 'periodic_update_task'):
#             self.periodic_update_task.cancel()
#
#     async def periodic_account_update(self):
#         while True:
#             await self.get_futures_balance()
#             await self.get_futures_positions()
#             await asyncio.sleep(5)  # Wait for 5 seconds before the next update
#
#
#     async def sync_server_time(self):
#         try:
#             server_time = await self.client.get_server_time()
#             self.client.timestamp_offset = server_time['serverTime'] - int(time.time() * 1000)
#         except Exception as e:
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': f"Error syncing server time: {str(e)}"
#             }))
#
#     async def start_user_socket(self):
#         if self.reconnecting:
#             return
#         self.reconnecting = True
#         self.user_socket = self.bm.futures_user_socket()
#         asyncio.create_task(self.user_socket_listener())
#
#     async def start_twm_in_thread(self):
#         def run_twm():
#             self.twm.start()
#             self.mark_price_socket = self.twm.start_all_mark_price_socket(
#                 callback=self.handle_mark_price_update,
#                 fast=True
#             )
#
#         Thread(target=run_twm).start()
#
#     async def initial_data_fetch(self):
#         await self.get_futures_balance()
#         await self.get_futures_positions()
#
#     async def user_socket_listener(self):
#         async with self.user_socket as tscm:
#             while True:
#                 try:
#                     res = await tscm.recv()
#                     if res:
#                         event_type = res.get('e')
#                         if event_type == 'ACCOUNT_UPDATE':
#                             await self.handle_account_update(res)
#                 except Exception as e:
#                     print(f"Error in user socket: {e}")
#                     await asyncio.sleep(5)
#                     await self.start_user_socket()
#                     break
#
#     async def mark_price_socket_listener(self):
#         async with self.mark_price_socket as mps:
#             while True:
#                 try:
#                     res = await mps.recv()
#                     if res:
#                         self.handle_mark_price_update(res)
#                 except Exception as e:
#                     print(f"Error in mark price socket: {e}")
#                     await asyncio.sleep(5)
#                     await self.start_mark_price_socket()
#                     break
#
#     async def handle_account_update(self, data):
#         balances = data['a']['B']
#         positions = data['a']['P']
#
#         usdt_balance = next((b for b in balances if b['a'] == 'USDT'), None)
#         if usdt_balance:
#             await self.send_if_changed('futures_balance', {
#                 'asset': 'USDT',
#                 'balance': usdt_balance['wb'],
#                 'crossWalletBalance': usdt_balance['cw'],  # cross wallet balance
#                 'availableBalance': usdt_balance['ab']  # available balance
#             })
#
#         active_positions = [position for position in positions if float(position['pa']) != 0]
#         if active_positions:
#             positions_data = []
#             for position in active_positions:
#                 position_data = {
#                     'symbol': position['s'],
#                     'positionAmt': position['pa'],
#                     'entryPrice': position['ep'],
#                     'unrealizedProfit': position['up'],
#                     'leverage': position['l'],
#                     'markPrice': self.mark_prices.get(position['s'], position['mp'])
#                 }
#                 position_data['profit_percentage'] = self.calculate_profit_percentage(position_data)
#                 positions_data.append(position_data)
#
#             await self.send_if_changed('futures_positions', positions_data)
#
#     def handle_mark_price_update(self, msg):
#         asyncio.run_coroutine_threadsafe(self._handle_mark_price_update(msg), asyncio.get_event_loop())
#
#
#     async def _handle_mark_price_update(self, msg):
#         symbol = msg['s']
#         mark_price = msg['p']
#         self.mark_prices[symbol] = mark_price
#         await self.update_positions_with_new_mark_price(symbol, mark_price)
#
#
#     async def update_positions_with_new_mark_price(self, symbol, mark_price):
#         if 'futures_positions' in self.last_sent_data:
#             positions = self.last_sent_data['futures_positions']
#             updated = False
#             for pos in positions:
#                 if pos['symbol'] == symbol:
#                     pos['markPrice'] = mark_price
#                     pos['profit_percentage'] = self.calculate_profit_percentage(pos)
#                     updated = True
#             if updated:
#                 await self.send_if_changed('futures_positions', positions)
#
#     async def send_if_changed(self, data_type, data):
#         key = data_type
#         if self.last_sent_data.get(key) != data:
#             self.last_sent_data[key] = data
#             await self.send(text_data=json.dumps({
#                 'type': data_type,
#                 'data': data
#             }))
#
#     def calculate_profit_percentage(self, position):
#         try:
#             position_amt = Decimal(position.get('positionAmt', '0'))
#             if position_amt == Decimal('0'):
#                 return Decimal('0')
#
#             entry_price = Decimal(position.get('entryPrice', '0'))
#             mark_price = Decimal(position.get('markPrice', '0'))
#             leverage = Decimal(position.get('leverage', '1'))
#
#             if entry_price == Decimal('0'):
#                 return Decimal('0')
#
#             if position_amt > Decimal('0'):  # Long position
#                 profit_percentage = ((mark_price - entry_price) / entry_price) * 100 * leverage
#             else:  # Short position
#                 profit_percentage = ((entry_price - mark_price) / entry_price) * 100 * leverage
#
#             return float(profit_percentage.quantize(Decimal('0.01')))
#         except (InvalidOperation, ZeroDivisionError):
#             return 0
#
#     async def get_futures_balance(self):
#         try:
#             futures_balances = await self.client.futures_account_balance()
#             futures_usdt_balance = next((item for item in futures_balances if item["asset"] == "USDT"), None)
#             await self.send_if_changed('futures_balance', futures_usdt_balance)
#         except BinanceAPIException as e:
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': f"Binance API Error: {str(e)}"
#             }))
#         except Exception as e:
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': f"Error fetching futures balance: {str(e)}"
#             }))
#
#     async def get_futures_positions(self):
#         try:
#             all_positions = await self.client.futures_position_information()
#             active_positions = [
#                 pos for pos in all_positions
#                 if float(pos["positionAmt"]) != 0
#             ]
#
#             positions_data = []
#             for pos in active_positions:
#                 pos_data = {
#                     'symbol': pos['symbol'],
#                     'positionAmt': pos['positionAmt'],
#                     'entryPrice': pos['entryPrice'],
#                     'markPrice': self.mark_prices.get(pos['symbol'], pos['markPrice']),
#                     'unRealizedProfit': pos['unRealizedProfit'],
#                     'liquidationPrice': pos['liquidationPrice'],
#                     'leverage': pos['leverage'],
#                 }
#                 pos_data['profit_percentage'] = self.calculate_profit_percentage(pos_data)
#                 positions_data.append(pos_data)
#
#             await self.send_if_changed('futures_positions', positions_data)
#
#         except BinanceAPIException as e:
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': f"Binance API Error: {str(e)}"
#             }))
#         except Exception as e:
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': f"Error fetching futures positions: {str(e)}"
#             }))
#
#     async def receive(self, text_data):
#         try:
#             data = json.loads(text_data)
#             if data['type'] == 'get_futures_positions':
#                 await self.get_futures_positions()
#             elif data['type'] == 'get_futures_balance':
#                 await self.get_futures_balance()
#         except json.JSONDecodeError:
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': 'Invalid JSON'
#             }))
#         except Exception as e:
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': str(e)
#             }))

# class UserDataConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         print("UserData WebSocket 연결 시도")
#         await self.accept()
#         print("UserData WebSocket 연결 성공")
#         self.client = await AsyncClient.create(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
#         self.bm = BinanceSocketManager(self.client)
#         self.user_socket = None
#         self.reconnecting = False
#         self.last_sent_data = {}
#         await self.start_user_socket()
#
#     async def disconnect(self, close_code):
#         print("UserData WebSocket 연결 종료")
#         if self.user_socket:
#             await self.user_socket.__aexit__(None, None, None)
#         if hasattr(self, 'client'):
#             await self.client.close_connection()
#         self.reconnecting = False
#
#     async def receive(self, text_data):
#         data = json.loads(text_data)
#         if data['type'] == 'get_futures_balance':
#             await self.send_futures_balance()
#         elif data['type'] == 'get_futures_positions':
#             await self.send_futures_positions()
#
#     async def start_user_socket(self):
#         print("사용자 소켓 시작")
#         if self.reconnecting:
#             return
#         self.reconnecting = True
#         self.user_socket = self.bm.futures_user_socket()
#         asyncio.create_task(self.user_socket_listener())
#
#
#     async def user_socket_listener(self):
#         print("사용자 소켓 리스너 시작")
#         async with self.user_socket as tscm:
#             while True:
#                 try:
#                     res = await tscm.recv()
#                     if res:
#                         print(f"수신된 데이터: {res}")
#                         event_type = res.get('e')
#                         if event_type == 'ACCOUNT_UPDATE':
#                             await self.handle_account_update(res)
#                 except Exception as e:
#                     print(f"Error in user socket: {e}")
#                     await asyncio.sleep(5)
#                     await self.start_user_socket()
#                     break
#     async def handle_account_update(self, data):
#         balances = data['a']['B']
#         positions = data['a']['P']
#
#         usdt_balance = next((b for b in balances if b['a'] == 'USDT'), None)
#         if usdt_balance:
#             await self.send_if_changed('futures_balance', {
#                 'asset': 'USDT',
#                 'balance': usdt_balance['wb'],
#                 'crossWalletBalance': usdt_balance.get('cw', '0'),
#                 'availableBalance': usdt_balance.get('ab', '0')
#             })
#
#         active_positions = [position for position in positions if float(position['pa']) != 0]
#         if active_positions:
#             positions_data = []
#             for position in active_positions:
#                 position_data = {
#                     'symbol': position['s'],
#                     'positionAmt': position['pa'],
#                     'entryPrice': position['ep'],
#                     'unrealizedProfit': position['up'],
#                     'leverage': position['l'],
#                     'markPrice': position['mp']
#                 }
#                 position_data['profit_percentage'] = self.calculate_profit_percentage(position_data)
#                 positions_data.append(position_data)
#
#             await self.send_if_changed('futures_positions', positions_data)
#
#     async def send_if_changed(self, data_type, data):
#         key = data_type
#         if self.last_sent_data.get(key) != data:
#             self.last_sent_data[key] = data
#             await self.send(text_data=json.dumps({
#                 'type': data_type,
#                 'data': data
#             }))
#             print(f"데이터 전송: {data_type} - {data}")
#
#     async def send_futures_balance(self):
#         print("선물 잔고 전송 시도")
#         if 'futures_balance' in self.last_sent_data:
#             await self.send(text_data=json.dumps({
#                 'type': 'futures_balance',
#                 'data': self.last_sent_data['futures_balance']
#             }))
#         else:
#             print("선물 잔고 데이터 없음")
#
#     async def send_futures_positions(self):
#         print("선물 포지션 전송 시도")
#         if 'futures_positions' in self.last_sent_data:
#             await self.send(text_data=json.dumps({
#                 'type': 'futures_positions',
#                 'data': self.last_sent_data['futures_positions']
#             }))
#         else:
#             print("선물 포지션 데이터 없음")
#
#     def calculate_profit_percentage(self, position):
#         try:
#             position_amt = Decimal(position.get('positionAmt', '0'))
#             if position_amt == Decimal('0'):
#                 return Decimal('0')
#
#             entry_price = Decimal(position.get('entryPrice', '0'))
#             mark_price = Decimal(position.get('markPrice', '0'))
#             leverage = Decimal(position.get('leverage', '1'))
#
#             if entry_price == Decimal('0'):
#                 return Decimal('0')
#
#             if position_amt > Decimal('0'):  # Long position
#                 profit_percentage = ((mark_price - entry_price) / entry_price) * 100 * leverage
#             else:  # Short position
#                 profit_percentage = ((entry_price - mark_price) / entry_price) * 100 * leverage
#
#             return float(profit_percentage.quantize(Decimal('0.01')))
#         except (InvalidOperation, ZeroDivisionError):
#             return 0
#
# class MarkPriceConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         print("MarkPrice WebSocket 연결 시도")
#         await self.accept()
#         print("MarkPrice WebSocket 연결 성공")
#         self.twm = ThreadedWebsocketManager(api_key=settings.BINANCE_API_KEY, api_secret=settings.BINANCE_API_SECRET)
#         self.mark_price_socket = None
#         self.mark_prices = {}
#         await self.start_twm()
#
#     async def disconnect(self, close_code):
#         print("MarkPrice WebSocket 연결 종료")
#         if self.mark_price_socket:
#             self.twm.stop_socket(self.mark_price_socket)
#         self.twm.stop()
#
#     async def start_twm(self):
#         loop = asyncio.get_event_loop()
#         await loop.run_in_executor(None, self.twm.start)
#         await self.start_all_mark_price_socket()
#
#     async def start_all_mark_price_socket(self):
#         print("마크 가격 소켓 시작")
#         loop = asyncio.get_event_loop()
#         self.mark_price_socket = await loop.run_in_executor(
#             None,
#             self.twm.start_all_mark_price_socket,
#             self.handle_mark_price_update,
#             True
#         )
#
#     def handle_mark_price_update(self, msg):
#         asyncio.run_coroutine_threadsafe(self._handle_mark_price_update(msg), asyncio.get_event_loop())
#
#     async def _handle_mark_price_update(self, msg):
#         try:
#             if isinstance(msg, str):
#                 msg = json.loads(msg)
#
#             if 'data' in msg and isinstance(msg['data'], list):
#                 for item in msg['data']:
#                     await self.process_mark_price_data(item)
#             else:
#                 print(f"Unexpected message format: {msg}")
#
#             await self.send_mark_prices()
#         except Exception as e:
#             print(f"Error in handle_mark_price_update: {e}")
#
#     async def process_mark_price_data(self, data):
#         symbol = data.get('s')
#         mark_price = data.get('p')
#         if symbol and mark_price:
#             self.mark_prices[symbol] = mark_price
#             print(f"Updated mark price for {symbol}: {mark_price}")
#         else:
#             print(f"Unable to extract mark price from: {data}")
#
#     async def send_mark_prices(self):
#         await self.send(text_data=json.dumps({
#             'type': 'mark_prices',
#             'data': self.mark_prices
#         }))


# class BinanceBaseConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         await self.accept()
#         self.client = await AsyncClient.create(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
#         self.bm = BinanceSocketManager(self.client)
#         self.twm = ThreadedWebsocketManager(api_key=settings.BINANCE_API_KEY, api_secret=settings.BINANCE_API_SECRET)
#         self.user_socket = None
#         self.mark_price_socket = None
#         self.reconnecting = False
#         self.last_sent_data = {}
#         self.mark_prices = {}
#         await self.sync_server_time()
#         await self.start_user_socket()
#         self.start_twm_in_thread()
#
#     def start_twm_in_thread(self):
#         def run_twm():
#             asyncio.set_event_loop(asyncio.new_event_loop())
#             self.twm.start()
#             self.start_all_mark_price_socket()
#
#         Thread(target=run_twm).start()
#
#     async def disconnect(self, close_code):
#         if self.user_socket:
#             await self.user_socket.close()
#         if self.mark_price_socket:
#             self.twm.stop_socket(self.mark_price_socket)
#         self.twm.stop()
#         if hasattr(self, 'client'):
#             await self.client.close_connection()
#         self.reconnecting = False
#
#     async def sync_server_time(self):
#         try:
#             server_time = await self.client.get_server_time()
#             self.client.timestamp_offset = server_time['serverTime'] - int(time.time() * 1000)
#         except Exception as e:
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': f"Error syncing server time: {str(e)}"
#             }))
#
#     async def start_user_socket(self):
#         if self.reconnecting:
#             return
#         self.reconnecting = True
#         self.user_socket = self.bm.futures_user_socket()
#         asyncio.create_task(self.user_socket_listener())
#
#     def start_all_mark_price_socket(self):
#         self.mark_price_socket = self.twm.start_all_mark_price_socket(
#             callback=self.handle_mark_price_update,
#             fast=True
#         )
#
#     async def user_socket_listener(self):
#         async with self.user_socket as tscm:
#             while True:
#                 try:
#                     res = await tscm.recv()
#                     if res:
#                         event_type = res.get('e')
#                         if event_type == 'ACCOUNT_UPDATE':
#                             await self.handle_account_update(res)
#                 except Exception as e:
#                     print(f"Error in user socket: {e}")
#                     await asyncio.sleep(5)
#                     await self.start_user_socket()
#                     break
#
#     async def handle_account_update(self, data):
#         balances = data['a']['B']
#         positions = data['a']['P']
#
#         usdt_balance = next((b for b in balances if b['a'] == 'USDT'), None)
#         if usdt_balance:
#             await self.send_if_changed('futures_balance', {
#                 'asset': 'USDT',
#                 'balance': usdt_balance['wb']
#             })
#
#         active_positions = [position for position in positions if float(position['pa']) != 0]
#         if active_positions:
#             positions_data = []
#             for position in active_positions:
#                 position_data = {
#                     'symbol': position['s'],
#                     'positionAmt': position['pa'],
#                     'entryPrice': position['ep'],
#                     'unrealizedProfit': position['up'],
#                     'leverage': position['l'],
#                     'markPrice': self.mark_prices.get(position['s'], position['mp'])
#                 }
#                 position_data['profit_percentage'] = self.calculate_profit_percentage(position_data)
#                 positions_data.append(position_data)
#
#             await self.send_if_changed('futures_positions', positions_data)
#
#     def handle_mark_price_update(self, msg):
#         for item in msg:
#             symbol = item['s']
#             mark_price = item['p']
#             self.mark_prices[symbol] = mark_price
#
#         asyncio.run_coroutine_threadsafe(self.update_positions_with_new_mark_price(), asyncio.get_event_loop())
#
#     async def update_positions_with_new_mark_price(self):
#         try:
#             all_positions = await self.client.futures_position_information()
#             active_positions = [
#                 pos for pos in all_positions
#                 if float(pos["positionAmt"]) != 0
#             ]
#
#             positions_data = []
#             for pos in active_positions:
#                 pos_data = {
#                     'symbol': pos['symbol'],
#                     'positionAmt': pos['positionAmt'],
#                     'entryPrice': pos['entryPrice'],
#                     'markPrice': self.mark_prices.get(pos['symbol'], pos['markPrice']),
#                     'unRealizedProfit': pos['unRealizedProfit'],
#                     'liquidationPrice': pos['liquidationPrice'],
#                     'leverage': pos['leverage'],
#                 }
#                 pos_data['profit_percentage'] = self.calculate_profit_percentage(pos_data)
#                 positions_data.append(pos_data)
#
#             await self.send_if_changed('futures_positions', positions_data)
#
#         except Exception as e:
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': str(e)
#             }))
#
#     async def send_if_changed(self, data_type, data):
#         key = data_type
#         if self.last_sent_data.get(key) != data:
#             self.last_sent_data[key] = data
#             await self.send(text_data=json.dumps({
#                 'type': data_type,
#                 'data': data
#             }))
#
#     def calculate_profit_percentage(self, position):
#         try:
#             position_amt = Decimal(position.get('positionAmt', '0'))
#             if position_amt == Decimal('0'):
#                 return Decimal('0')
#
#             entry_price = Decimal(position.get('entryPrice', '0'))
#             mark_price = Decimal(position.get('markPrice', '0'))
#             leverage = Decimal(position.get('leverage', '1'))
#
#             if entry_price == Decimal('0'):
#                 return Decimal('0')
#
#             if position_amt > Decimal('0'):  # Long position
#                 profit_percentage = ((mark_price - entry_price) / entry_price) * 100 * leverage
#             else:  # Short position
#                 profit_percentage = ((entry_price - mark_price) / entry_price) * 100 * leverage
#
#             return float(profit_percentage.quantize(Decimal('0.01')))
#         except (InvalidOperation, ZeroDivisionError):
#             return 0
#
#
# class PeriodicDataConsumer(BinanceBaseConsumer):
#     async def connect(self):
#         await super().connect()
#         self.periodic_task = asyncio.create_task(self.periodically_send_data())
#
#     async def disconnect(self, close_code):
#         if hasattr(self, 'periodic_task'):
#             self.periodic_task.cancel()
#         await super().disconnect(close_code)
#
#     async def periodically_send_data(self):
#         while True:
#             try:
#                 await self.get_futures_balance()
#                 await self.get_futures_positions()
#                 await asyncio.sleep(5)  # Send data every 1 second
#             except asyncio.CancelledError:
#                 break
#             except Exception as e:
#                 print(f"Error in periodic task: {e}")
#                 await asyncio.sleep(1)
#
#     async def get_futures_balance(self):
#         try:
#             futures_balances = await self.client.futures_account_balance()
#             futures_usdt_balance = next((item for item in futures_balances if item["asset"] == "USDT"), None)
#             await self.send_if_changed('futures_balance', futures_usdt_balance)
#         except Exception as e:
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': str(e)
#             }))
#
#     async def get_futures_positions(self):
#         try:
#             all_positions = await self.client.futures_position_information()
#             active_positions = [
#                 pos for pos in all_positions
#                 if float(pos["positionAmt"]) != 0
#             ]
#
#             positions_data = []
#             for pos in active_positions:
#                 pos_data = {
#                     'symbol': pos['symbol'],
#                     'positionAmt': pos['positionAmt'],
#                     'entryPrice': pos['entryPrice'],
#                     'markPrice': self.mark_prices.get(pos['symbol'], pos['markPrice']),
#                     'unRealizedProfit': pos['unRealizedProfit'],
#                     'liquidationPrice': pos['liquidationPrice'],
#                     'leverage': pos['leverage'],
#                 }
#                 pos_data['profit_percentage'] = self.calculate_profit_percentage(pos_data)
#                 positions_data.append(pos_data)
#
#             await self.send_if_changed('futures_positions', positions_data)
#
#         except Exception as e:
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': str(e)
#             }))
#
#     async def receive(self, text_data):
#         try:
#             data = json.loads(text_data)
#             if data['type'] == 'get_futures_positions':
#                 await self.get_futures_positions()
#             elif data['type'] == 'get_futures_balance':
#                 await self.get_futures_balance()
#         except json.JSONDecodeError:
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': 'Invalid JSON'
#             }))
#         except Exception as e:
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': str(e)
#             }))
