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


# class BinanceClient:
#     def __init__(self):
#         self.base_url = 'wss://fstream.binance.com/ws/'
#         self.api_url = 'https://fapi.binance.com'
#         self.api_key = settings.BINANCE_API_KEY
#         self.api_secret = settings.BINANCE_API_SECRET
#         self.callbacks = {}
#         self.user_ws = None
#
#     def connect(self):
#         listen_key = self.get_listen_key()
#
#         # User Data Stream 연결
#         user_socket_url = f"{self.base_url}{listen_key}"
#         self.user_ws = websocket.WebSocketApp(
#             user_socket_url,
#             on_message=self.on_user_message,
#             on_error=self.on_error,
#             on_close=self.on_close
#         )
#         user_wst = threading.Thread(target=self.user_ws.run_forever)
#         user_wst.daemon = True
#         user_wst.start()
#
#     def get_listen_key(self):
#         endpoint = f"{self.api_url}/fapi/v1/listenKey"
#         headers = {"X-MBX-APIKEY": self.api_key}
#         response = requests.post(endpoint, headers=headers)
#         data = response.json()
#         return data['listenKey']
#
#     def on_user_message(self, ws, message):
#         data = json.loads(message)
#         event_type = data.get('e')
#         if event_type in self.callbacks:
#             self.callbacks[event_type](data)
#
#     def on_error(self, ws, error):
#         print(f"Error: {error}")
#
#     def on_close(self, ws, close_status_code, close_msg):
#         print("WebSocket connection closed")
#
#     def add_callback(self, event_type, callback):
#         self.callbacks[event_type] = callback
#
#     def close(self):
#         if self.user_ws:
#             self.user_ws.close()
#
#     # 초기 데이터 로딩을 위한 REST API 호출 메서드
#     def get_initial_data(self):
#         balance_endpoint = f"{self.api_url}/fapi/v2/balance"
#         position_endpoint = f"{self.api_url}/fapi/v2/positionRisk"
#
#         balance_data = self.send_signed_request('GET', balance_endpoint)
#         position_data = self.send_signed_request('GET', position_endpoint)
#
#         return {
#             'balance': balance_data,
#             'positions': position_data
#         }
#
#     def send_signed_request(self, method, url_path, payload=None):
#         if payload is None:
#             payload = {}
#         payload['timestamp'] = int(time.time() * 1000)
#         payload['recvWindow'] = 60000  # 최대 허용 시간
#         query_string = '&'.join([f"{k}={v}" for k, v in payload.items()])
#         signature = hmac.new(self.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
#         payload['signature'] = signature
#         headers = {
#             'X-MBX-APIKEY': self.api_key
#         }
#         response = requests.request(method, f"{url_path}?{query_string}", headers=headers)
#         return response.json()
#
#
# class BinanceDataConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         await self.accept()
#         self.binance_client = BinanceClient()
#         self.binance_client.add_callback('ACCOUNT_UPDATE', self.handle_account_update)
#
#         # 초기 데이터 가져오기
#         await self.fetch_initial_data()
#
#         # 웹소켓 연결 시작
#         self.binance_client.connect()
#
#     async def disconnect(self, close_code):
#         self.binance_client.close()
#
#     async def receive(self, text_data):
#         # 클라이언트로부터의 요청은 무시하고 현재 저장된 데이터만 반환
#         data = json.loads(text_data)
#         if data['type'] == 'get_account_info':
#             await self.send_current_data()
#
#     async def fetch_initial_data(self):
#         # 초기 데이터를 한 번만 REST API를 통해 가져옵니다
#         initial_data = self.binance_client.get_initial_data()
#
#         print("Received initial data:", initial_data)  # 디버깅을 위한 로그
#
#         try:
#             # balance 데이터 처리
#             if isinstance(initial_data['balance'], list):
#                 self.current_balance = next((b for b in initial_data['balance'] if b['asset'] == 'USDT'), None)
#             elif isinstance(initial_data['balance'], dict):
#                 self.current_balance = initial_data['balance'].get('USDT')
#             else:
#                 print("Unexpected balance data format:", initial_data['balance'])
#                 self.current_balance = None
#
#             # positions 데이터 처리
#             if isinstance(initial_data['positions'], list):
#                 self.current_positions = [p for p in initial_data['positions'] if float(p.get('positionAmt', '0')) != 0]
#             else:
#                 print("Unexpected positions data format:", initial_data['positions'])
#                 self.current_positions = []
#
#             if self.current_balance:
#                 self.current_balance = {
#                     'asset': 'USDT',
#                     'balance': self.current_balance.get('balance', '0'),
#                     'crossWalletBalance': self.current_balance.get('crossWalletBalance', '0'),
#                     'availableBalance': self.current_balance.get('availableBalance', '0')
#                 }
#
#         except Exception as e:
#             print(f"Error processing initial data: {e}")
#             self.current_balance = None
#             self.current_positions = []
#
#         await self.send_current_data()
#
#     async def handle_account_update(self, data):
#         # User Data Stream으로부터 받은 계정 업데이트 처리
#         if 'a' in data:
#             balances = data['a'].get('B', [])
#             positions = data['a'].get('P', [])
#
#             usdt_balance = next((b for b in balances if b['a'] == 'USDT'), None)
#             if usdt_balance:
#                 self.current_balance = {
#                     'asset': 'USDT',
#                     'balance': usdt_balance['wb'],  # wallet balance
#                     'crossWalletBalance': usdt_balance['cw'],  # cross wallet balance
#                     'availableBalance': usdt_balance['ab']  # available balance
#                 }
#
#             if positions:
#                 self.current_positions = [p for p in positions if float(p['pa']) != 0]
#
#             await self.send_current_data()
#
#     async def send_current_data(self):
#         await self.send(text_data=json.dumps({
#             'type': 'account_update',
#             'data': {
#                 'balance': self.current_balance,
#                 'positions': self.process_positions(self.current_positions)
#             }
#         }))
#
#
#     def process_positions(self, positions):
#         positions_data = []
#         for pos in positions:
#             pos_data = {
#                 'symbol': pos['s'],
#                 'positionAmt': pos['pa'],
#                 'entryPrice': pos['ep'],
#                 'markPrice': pos['mp'],
#                 'unRealizedProfit': pos['up'],
#                 'liquidationPrice': pos['lp'],
#                 'leverage': pos['l'],
#             }
#             pos_data['profit_percentage'] = self.calculate_profit_percentage(pos_data)
#             positions_data.append(pos_data)
#         return positions_data
#
#     def calculate_profit_percentage(self, position):
#         try:
#             position_amt = Decimal(position['positionAmt'])
#             if position_amt == Decimal('0'):
#                 return Decimal('0')
#
#             entry_price = Decimal(position['entryPrice'])
#             mark_price = Decimal(position['markPrice'])
#             leverage = Decimal(position['leverage'])
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


class BinanceWebSocketConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.client = await AsyncClient.create(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
        self.bm = BinanceSocketManager(self.client)
        self.twm = ThreadedWebsocketManager(api_key=settings.BINANCE_API_KEY, api_secret=settings.BINANCE_API_SECRET)

        self.user_socket = None
        self.mark_price_socket = None
        self.reconnecting = False
        self.last_sent_data = {}
        self.mark_prices = {}
        await self.sync_server_time()
        await self.start_user_socket()
        await self.start_twm_in_thread()
        await self.initial_data_fetch()

    async def disconnect(self, close_code):
        if self.user_socket:
            await self.user_socket.__aexit__(None, None, None)
        if self.twm:
            self.twm.stop()
        if hasattr(self, 'client'):
            await self.client.close_connection()
        self.reconnecting = False

    async def sync_server_time(self):
        try:
            server_time = await self.client.get_server_time()
            self.client.timestamp_offset = server_time['serverTime'] - int(time.time() * 1000)
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error syncing server time: {str(e)}"
            }))

    async def start_user_socket(self):
        if self.reconnecting:
            return
        self.reconnecting = True
        self.user_socket = self.bm.futures_user_socket()
        asyncio.create_task(self.user_socket_listener())

    async def start_twm_in_thread(self):
        def run_twm():
            self.twm.start()
            self.mark_price_socket = self.twm.start_all_mark_price_socket(
                callback=self.handle_mark_price_update,
                fast=True
            )

        Thread(target=run_twm).start()
    def start_all_mark_price_socket(self):
        self.mark_price_socket = self.twm.start_all_mark_price_socket(
            callback=self.handle_mark_price_update,
            fast=True
        )
    async def initial_data_fetch(self):
        await self.get_futures_balance()
        await self.get_futures_positions()

    async def user_socket_listener(self):
        async with self.user_socket as tscm:
            while True:
                try:
                    res = await tscm.recv()
                    if res:
                        event_type = res.get('e')
                        if event_type == 'ACCOUNT_UPDATE':
                            await self.handle_account_update(res)
                except Exception as e:
                    print(f"Error in user socket: {e}")
                    await asyncio.sleep(5)
                    await self.start_user_socket()
                    break

    async def mark_price_socket_listener(self):
        async with self.mark_price_socket as mps:
            while True:
                try:
                    res = await mps.recv()
                    if res:
                        self.handle_mark_price_update(res)
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
                'crossWalletBalance': usdt_balance['cw'],  # cross wallet balance
                'availableBalance': usdt_balance['ab']  # available balance
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

    def handle_mark_price_update(self, msg):
        asyncio.run_coroutine_threadsafe(self._handle_mark_price_update(msg), asyncio.get_event_loop())


    async def _handle_mark_price_update(self, msg):
        symbol = msg['s']
        mark_price = msg['p']
        self.mark_prices[symbol] = mark_price
        await self.update_positions_with_new_mark_price(symbol, mark_price)


    async def update_positions_with_new_mark_price(self, symbol, mark_price):
        if 'futures_positions' in self.last_sent_data:
            positions = self.last_sent_data['futures_positions']
            updated = False
            for pos in positions:
                if pos['symbol'] == symbol:
                    pos['markPrice'] = mark_price
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

            if position_amt > Decimal('0'):  # Long position
                profit_percentage = ((mark_price - entry_price) / entry_price) * 100 * leverage
            else:  # Short position
                profit_percentage = ((entry_price - mark_price) / entry_price) * 100 * leverage

            return float(profit_percentage.quantize(Decimal('0.01')))
        except (InvalidOperation, ZeroDivisionError):
            return 0

    async def get_futures_balance(self):
        try:
            futures_balances = await self.client.futures_account_balance()
            futures_usdt_balance = next((item for item in futures_balances if item["asset"] == "USDT"), None)
            await self.send_if_changed('futures_balance', futures_usdt_balance)
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    async def get_futures_positions(self):
        try:
            all_positions = await self.client.futures_position_information()
            active_positions = [
                pos for pos in all_positions
                if float(pos["positionAmt"]) != 0
            ]

            positions_data = []
            for pos in active_positions:
                pos_data = {
                    'symbol': pos['symbol'],
                    'positionAmt': pos['positionAmt'],
                    'entryPrice': pos['entryPrice'],
                    'markPrice': self.mark_prices.get(pos['symbol'], pos['markPrice']),
                    'unRealizedProfit': pos['unRealizedProfit'],
                    'liquidationPrice': pos['liquidationPrice'],
                    'leverage': pos['leverage'],
                }
                pos_data['profit_percentage'] = self.calculate_profit_percentage(pos_data)
                positions_data.append(pos_data)

            await self.send_if_changed('futures_positions', positions_data)

        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            if data['type'] == 'get_futures_positions':
                await self.get_futures_positions()
            elif data['type'] == 'get_futures_balance':
                await self.get_futures_balance()
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Invalid JSON'
            }))
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))

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
