import asyncio
from django.conf import settings
import time
from decimal import Decimal, InvalidOperation
from binance import AsyncClient, BinanceSocketManager, ThreadedWebsocketManager
from channels.generic.websocket import AsyncWebsocketConsumer
import json
from threading import Thread
class PeriodicDataConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print("WebSocket 연결 시도")
        await self.accept()
        print("WebSocket 연결 성공")
        self.client = await AsyncClient.create(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
        self.bm = BinanceSocketManager(self.client)
        self.twm = ThreadedWebsocketManager(api_key=settings.BINANCE_API_KEY, api_secret=settings.BINANCE_API_SECRET)
        self.user_socket = None
        self.mark_price_socket = None
        self.reconnecting = False
        self.last_sent_data = {}
        self.mark_prices = {}
        await self.start_user_socket()
        self.start_twm_in_thread()

    # def start_twm_in_thread(self):
    #     def run_twm():
    #         asyncio.set_event_loop(asyncio.new_event_loop())
    #         self.twm.start()
    #         self.start_all_mark_price_socket()
    #
    #     Thread(target=run_twm).start()

    def start_twm_in_thread(self):
        def run_twm():
            loop = asyncio.new_event_loop()  # Create a new event loop
            asyncio.set_event_loop(loop)  # Set this loop as the current one for this thread
            self.twm.start()
            loop.run_until_complete(self.start_all_mark_price_socket())
            loop.run_forever()

        self.thread = Thread(target=run_twm)
        self.thread.start()

    async def disconnect(self, close_code):
        print("WebSocket 연결 종료")
        if self.user_socket:
            # Try manually closing the websocket
            await self.user_socket.__aexit__(None, None, None)
        if self.mark_price_socket:
            self.twm.stop_socket(self.mark_price_socket)
        self.twm.stop()
        if hasattr(self, 'client'):
            await self.client.close_connection()
        self.reconnecting = False

    async def receive(self, text_data):
        data = json.loads(text_data)
        if data['type'] == 'get_futures_balance':
            await self.send_futures_balance()
        elif data['type'] == 'get_futures_positions':
            await self.send_futures_positions()

    async def start_user_socket(self):
        print("사용자 소켓 시작")

        if self.reconnecting:
            return
        self.reconnecting = True
        self.user_socket = self.bm.futures_user_socket()
        asyncio.create_task(self.user_socket_listener())

    async def start_all_mark_price_socket(self):
        print("마크 가격 소켓 시작")

        self.mark_price_socket = self.twm.start_all_mark_price_socket(
            callback=self.handle_mark_price_update,
            fast=True
        )

    async def user_socket_listener(self):
        print("사용자 소켓 리스너 시작")

        async with self.user_socket as tscm:
            while True:
                try:
                    res = await tscm.recv()
                    if res:
                        print(f"수신된 데이터: {res}")

                        event_type = res.get('e')
                        if event_type == 'ACCOUNT_UPDATE':
                            await self.handle_account_update(res)
                except Exception as e:
                    print(f"Error in user socket: {e}")
                    await asyncio.sleep(5)
                    await self.start_user_socket()
                    break

    async def handle_account_update(self, data):
        balances = data['a']['B']
        positions = data['a']['P']

        usdt_balance = next((b for b in balances if b['a'] == 'USDT'), None)
        if usdt_balance:
            await self.send_if_changed('futures_balance', {
                'asset': 'USDT',
                'balance': usdt_balance['wb'],
                'crossWalletBalance': usdt_balance.get('cw', '0'),
                'availableBalance': usdt_balance.get('ab', '0')
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
        for item in msg:
            if isinstance(item, dict):  # Ensure that item is a dictionary
                symbol = item.get('s')  # Safely access the symbol key
                mark_price = item.get('p')
                if symbol and mark_price:
                    self.mark_prices[symbol] = mark_price
            else:
                print(f"Unexpected item format: {item}")  # Handle unexpected formats

        asyncio.run_coroutine_threadsafe(self.update_positions_with_new_mark_price(), asyncio.get_event_loop())

    # def handle_mark_price_update(self, msg):
    #     for item in msg:
    #         symbol = item['s']
    #         mark_price = item['p']
    #         self.mark_prices[symbol] = mark_price
    #
    #     # Instead of directly calling asyncio.run_coroutine_threadsafe, get the running loop:
    #     loop = asyncio.get_event_loop()
    #     if loop.is_running():
    #         asyncio.run_coroutine_threadsafe(self.update_positions_with_new_mark_price(), loop)
    #     else:
    #         # If there's no running loop, execute it normally
    #         loop.run_until_complete(self.update_positions_with_new_mark_price())

    async def update_positions_with_new_mark_price(self):
        try:
            positions_data = []
            for symbol, mark_price in self.mark_prices.items():
                position = next((pos for pos in self.last_sent_data.get('futures_positions', []) if pos['symbol'] == symbol), None)
                if position:
                    position['markPrice'] = mark_price
                    position['profit_percentage'] = self.calculate_profit_percentage(position)
                    positions_data.append(position)

            if positions_data:
                await self.send_if_changed('futures_positions', positions_data)

        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    async def send_if_changed(self, data_type, data):
        key = data_type
        if self.last_sent_data.get(key) != data:
            self.last_sent_data[key] = data
            await self.send(text_data=json.dumps({
                'type': data_type,
                'data': data
            }))

    async def send_futures_balance(self):
        print("선물 잔고 전송 시도")

        if 'futures_balance' in self.last_sent_data:
            await self.send(text_data=json.dumps({
                'type': 'futures_balance',
                'data': self.last_sent_data['futures_balance']
            }))

    async def send_futures_positions(self):
        print("선물 포지션 전송 시도")

        if 'futures_positions' in self.last_sent_data:
            await self.send(text_data=json.dumps({
                'type': 'futures_positions',
                'data': self.last_sent_data['futures_positions']
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

