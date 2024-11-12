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
from typing import Dict, Set, List, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from django.apps import apps
from asgiref.sync import sync_to_async
from datetime import datetime


class BinanceWebSocketConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.client = await AsyncClient.create(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
        self.bm = BinanceSocketManager(self.client)

        self.user_socket = None
        self.mark_price_socket = None
        self.last_sent_data = {'ACCOUNT_UPDATE': {'balance': None, 'positions': []}}
        self.mark_prices = {}
        # self.listen_key = None
        await self.request_account_update()  # 초기 데이터 로딩
        await self.start_user_data_stream()
        await self.start_mark_price_socket()
        # asyncio.create_task(self.keep_listen_key_alive())

        # 스케줄러 설정 (1시간마다 실행)
        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_job(
            self.hourly_account_snapshot,
            IntervalTrigger(hours=1)
        )
        self.scheduler.start()


    async def disconnect(self, close_code):
        if self.user_socket:
            await self.user_socket.__aexit__(None, None, None)
        if self.mark_price_socket:
            await self.mark_price_socket.__aexit__(None, None, None)
        if hasattr(self, 'client'):
            await self.client.close_connection()
        self.scheduler.shutdown()
        print("WebSocket connection closed and streams stopped.")

    async def start_user_data_stream(self):
        # self.listen_key = await self.client.futures_stream_get_listen_key()
        self.user_socket = self.bm.futures_user_socket()
        asyncio.create_task(self.user_socket_listener())

    async def start_mark_price_socket(self):

        # {'stream': '!markPrice@arr@1s', 'data': [
        #     {'e': 'markPriceUpdate', 'E': 1725086955001, 's': 'BTCUSDT', 'p': '59160.00000000', 'P': '59227.68354352',
        #      'i': '59186.34893617', 'r': '0.00001643', 'T': 1725091200000},
        #     {'e': 'markPriceUpdate', 'E': 1725086955001, 's': 'ETHUSDT', 'p': '2523.44000000', 'P': '2525.39100082',
        #      'i': '2524.47886364', 'r': '0.00006360', 'T': 1725091200000},
        self.mark_price_socket = self.bm.futures_multiplex_socket(['!markPrice@arr@1s'])
        asyncio.create_task(self.mark_price_socket_listener())
        print("Mark price socket started.")

    # async def keep_listen_key_alive(self):
    #     while True:
    #         await asyncio.sleep(30 * 60)  # 30 minutes
    #         try:
    #             await self.client.futures_stream_keepalive(self.listen_key)
    #         except Exception as e:
    #             print(f"Error keeping listen key alive: {e}")
    #             await self.start_user_data_stream()

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

    @sync_to_async
    def save_hourly_balance(self, balance_data, positions_data):
        HourlyBalance = apps.get_model('binanaceAccount', 'DailyBalance')
        new_balance = HourlyBalance.objects.create(
            futures_balance=balance_data,
            futures_positions=positions_data
        )
        return new_balance


    async def hourly_account_snapshot(self):
        try:
            current_data = self.last_sent_data['ACCOUNT_UPDATE']
            balance_data = current_data['balance']
            positions_data = current_data['positions']

            await self.save_hourly_balance(balance_data, positions_data)
            print(f"Hourly account snapshot saved at {datetime.now().isoformat()}")
        except Exception as e:
            print(f"Error saving hourly account snapshot: {str(e)}")


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

        # print(data)
        # print("-------------")
        balances = data['a']['B']
        positions = data['a']['P']

        usdt_balance = next((b for b in balances if b['a'] == 'USDT'), None)
        balance_data = None
        if usdt_balance:
            balance_data = {
                'asset': 'USDT',
                'balance': usdt_balance['wb'],
                'crossWalletBalance': usdt_balance['cw'],
                'availableBalance': usdt_balance['ab']
            }

        active_positions = [position for position in positions if float(position['pa']) != 0]
        positions_data = []
        if active_positions:
            for position in active_positions:
                position_data = {
                    'symbol': position['s'],
                    'positionAmt': position['pa'],
                    'entryPrice': position['ep'],
                    'unrealizedProfit': position['up'],
                    'leverage': position['l'],
                    'markPrice': self.mark_prices.get(position['s'], position['mp']),
                    'liquidationPrice': position.get('lp', '0')  # 청산가 추가
                }
                position_data['profit_percentage'] = self.calculate_profit_percentage(position_data)
                positions_data.append(position_data)
        # 'marginType': position.get('mt', ''),

        await self.send_if_changed('ACCOUNT_UPDATE', {
            'balance': balance_data,
            'positions': positions_data
        })

    async def handle_order_update(self, data):
        # {
        #     "e": "ORDER_TRADE_UPDATE",
        #     "T": 1725612509523,
        #     "E": 1725612509523,
        #     "o": {
        #         "s": "BTCUSDT",
        #         "c": "x-cLbi5uMH240905084314000087",
        #         "S": "SELL",
        #         "o": "LIMIT",
        #         "f": "GTC",
        #         "q": "0.002",
        #         "p": "57350",
        #         "ap": "0",
        #         "sp": "0",
        #         "x": "CANCELED",
        #         "X": "CANCELED",
        #         "i": 414013813089,
        #         "l": "0",
        #         "z": "0",
        #         "L": "0",
        #         "n": "0",
        #         "N": "USDT",
        #         "T": 1725612509523,
        #         "t": 0,
        #         "b": "436.67200",
        #         "a": "565.78600",
        #         "m": false,
        #         "R": false,
        #         "wt": "CONTRACT_PRICE",
        #         "ot": "LIMIT",
        #         "ps": "BOTH",
        #         "cp": false,
        #         "rp": "0",
        #         "pP": false,
        #         "si": 0,
        #         "ss": 0,
        #         "V": "NONE",
        #         "pm": "NONE",
        #         "gtd": 0
        #     }
        # }
        await self.send(text_data=json.dumps({
            'type': 'ORDER_TRADE_UPDATE',
            'data': data
        }))

    async def handle_mark_price_update(self, message):
        stream = message['stream']
        data = message['data']

        updated_symbols = set()
        active_symbols = set(
            pos['symbol'] for pos in self.last_sent_data.get('ACCOUNT_UPDATE', {}).get('positions', []))

        for item in data:
            symbol = item['s']
            price = item['p']
            if symbol in active_symbols:
                if symbol not in self.mark_prices or self.mark_prices[symbol] != price:
                    self.mark_prices[symbol] = price
                    updated_symbols.add(symbol)

        if updated_symbols:
            await self.update_positions_with_new_mark_prices(updated_symbols)

    async def update_positions_with_new_mark_prices(self, updated_symbols):
        account_data = self.last_sent_data.get('ACCOUNT_UPDATE', {})
        positions = account_data.get('positions', [])
        balance_data = account_data.get('balance', {})

        total_unrealized_profit = Decimal('0')

        for position in positions:
            if position['symbol'] in updated_symbols:
                old_mark_price = Decimal(position['markPrice'])
                new_mark_price = Decimal(self.mark_prices[position['symbol']])
                position_amt = Decimal(position['positionAmt'])
                entry_price = Decimal(position['entryPrice'])

                # Update markPrice
                position['markPrice'] = str(new_mark_price)

                # Recalculate unrealizedProfit
                old_unrealized_profit = Decimal(position['unrealizedProfit'])
                new_unrealized_profit = position_amt * (new_mark_price - entry_price)
                position['unrealizedProfit'] = str(new_unrealized_profit)

                # Update profit_percentage
                position['profit_percentage'] = self.calculate_profit_percentage(position)

                # Calculate the change in unrealized profit
                unrealized_profit_change = new_unrealized_profit - old_unrealized_profit
                total_unrealized_profit += unrealized_profit_change

        # Update balance data
        if balance_data:
            balance = Decimal(balance_data['balance'])
            cross_wallet_balance = Decimal(balance_data['crossWalletBalance'])
            available_balance = Decimal(balance_data['availableBalance'])

            balance_data['balance'] = str(balance + total_unrealized_profit)
            balance_data['crossWalletBalance'] = str(cross_wallet_balance)
            balance_data['availableBalance'] = str(available_balance)

        # Update the last sent data
        self.last_sent_data['ACCOUNT_UPDATE'] = {
            'balance': balance_data,
            'positions': positions
        }

        # Send updated account information to the client
        await self.send(text_data=json.dumps({
            'type': 'ACCOUNT_UPDATE',
            'data': self.last_sent_data['ACCOUNT_UPDATE']
        }))

    async def request_account_update(self):
        try:
            account_info = await self.client.futures_account()

            usdt_balance = next((b for b in account_info['assets'] if b['asset'] == 'USDT'), None)
            balance_data = None
            if usdt_balance:
                balance_data = {
                    'asset': 'USDT',
                    'balance': usdt_balance['walletBalance'],
                    'crossWalletBalance': usdt_balance['crossWalletBalance'],
                    'availableBalance': usdt_balance['availableBalance']
                }

            positions = account_info['positions']
            active_positions = [position for position in positions if float(position['positionAmt']) != 0]
            positions_data = []
            if active_positions:
                for position in active_positions:
                    position_data = {
                        'symbol': position['symbol'],
                        'positionAmt': position['positionAmt'],
                        'entryPrice': position['entryPrice'],
                        'unrealizedProfit': position['unrealizedProfit'],
                        'leverage': position['leverage'],
                        'markPrice': self.mark_prices.get(position['symbol'], position.get('markPrice', '0')),
                        'liquidationPrice': position.get('liquidationPrice', '0')  # 청산가 추가

                    }
                    position_data['profit_percentage'] = self.calculate_profit_percentage(position_data)
                    positions_data.append(position_data)

            print(balance_data)
            print("-------------------------------------")
            print(positions_data)
            await self.send_if_changed('ACCOUNT_UPDATE', {
                'balance': balance_data,
                'positions': positions_data
            })

        except Exception as e:
            print(f"Error requesting account update: {str(e)}")
            # print(f"Account info: {account_info}")  # 디버깅을 위해 전체 응답 출력
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f"Error requesting account update: {str(e)}"
            }))

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
#         self.client = await AsyncClient.create()
#         self.bm = BinanceSocketManager(self.client)
#
#         self.user_socket = None
#         self.mark_price_socket = None
#         self.last_sent_data = {'ACCOUNT_UPDATE': {'balance': None, 'positions': []}}
#         self.mark_prices = {}
#
#         await self.load_mock_data()  # 임의의 초기 데이터 로드
#         await self.start_user_data_stream()
#         await self.start_mark_price_socket()
#
#     async def disconnect(self, close_code):
#         if self.user_socket:
#             await self.user_socket.__aexit__(None, None, None)
#         if self.mark_price_socket:
#             await self.mark_price_socket.__aexit__(None, None, None)
#         if hasattr(self, 'client'):
#             await self.client.close_connection()
#
#     async def load_mock_data(self):
#         # 임의의 초기 데이터 생성
#         mock_balance = {
#             'asset': 'USDT',
#             'balance': '132.10',
#             'crossWalletBalance': '133.08',
#             'availableBalance': '132.10'
#         }
#         mock_positions = [
#             {
#                 'symbol': 'BTCUSDT',
#                 'positionAmt': '-0.006',
#                 'entryPrice': '59138.32',
#                 'unrealizedProfit': '-3.99',
#                 'leverage': '20',
#                 'markPrice': '55000',
#                 'liquidationPrice': '70952.26'
#             }
#         ]
#
#         for position in mock_positions:
#             position['profit_percentage'] = self.calculate_profit_percentage(position)
#             self.mark_prices[position['symbol']] = position['markPrice']
#
#         self.last_sent_data['ACCOUNT_UPDATE'] = {
#             'balance': mock_balance,
#             'positions': mock_positions
#         }
#
#         await self.send(text_data=json.dumps({
#             'type': 'ACCOUNT_UPDATE',
#             'data': self.last_sent_data['ACCOUNT_UPDATE']
#         }))
#
#     async def start_user_data_stream(self):
#         self.user_socket = self.bm.futures_user_socket()
#         asyncio.create_task(self.user_socket_listener())
#
#     async def start_mark_price_socket(self):
#         self.mark_price_socket = self.bm.futures_multiplex_socket(['!markPrice@arr@1s'])
#         asyncio.create_task(self.mark_price_socket_listener())
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
#                         elif event_type == 'ORDER_TRADE_UPDATE':
#                             await self.handle_order_update(res)
#                 except Exception as e:
#                     print(f"Error in user socket: {e}")
#                     await asyncio.sleep(5)
#                     await self.start_user_data_stream()
#                     break
#
#     async def mark_price_socket_listener(self):
#         async with self.mark_price_socket as mps:
#             while True:
#                 try:
#                     res = await mps.recv()
#                     if res:
#                         await self.handle_mark_price_update(res)
#                 except Exception as e:
#                     print(f"Error in mark price socket: {e}")
#                     await asyncio.sleep(5)
#                     await self.start_mark_price_socket()
#                     break
#
#     async def handle_account_update(self, data):
#
#         # print(data)
#         # print("-------------")
#         balances = data['a']['B']
#         positions = data['a']['P']
#
#         usdt_balance = next((b for b in balances if b['a'] == 'USDT'), None)
#         balance_data = None
#         if usdt_balance:
#             balance_data = {
#                 'asset': 'USDT',
#                 'balance': usdt_balance['wb'],
#                 'crossWalletBalance': usdt_balance['cw'],
#                 'availableBalance': usdt_balance['ab']
#             }
#
#         active_positions = [position for position in positions if float(position['pa']) != 0]
#         positions_data = []
#         if active_positions:
#             for position in active_positions:
#                 position_data = {
#                     'symbol': position['s'],
#                     'positionAmt': position['pa'],
#                     'entryPrice': position['ep'],
#                     'unrealizedProfit': position['up'],
#                     'leverage': position['l'],
#                     'markPrice': self.mark_prices.get(position['s'], position['mp']),
#                     'liquidationPrice': position.get('lp', '0')  # 청산가 추가
#                 }
#                 position_data['profit_percentage'] = self.calculate_profit_percentage(position_data)
#                 positions_data.append(position_data)
#         # 'marginType': position.get('mt', ''),
#
#         await self.send_if_changed('ACCOUNT_UPDATE', {
#             'balance': balance_data,
#             'positions': positions_data
#         })
#
#     async def handle_order_update(self, data):
#         await self.send(text_data=json.dumps({
#             'type': 'ORDER_TRADE_UPDATE',
#             'data': data
#         }))
#
#     async def handle_mark_price_update(self, message):
#         stream = message['stream']
#         data = message['data']
#
#         updated_symbols = set()
#         active_symbols = set(
#             pos['symbol'] for pos in self.last_sent_data.get('ACCOUNT_UPDATE', {}).get('positions', []))
#
#         for item in data:
#             symbol = item['s']
#             price = item['p']
#             if symbol in active_symbols:
#                 if symbol not in self.mark_prices or self.mark_prices[symbol] != price:
#                     self.mark_prices[symbol] = price
#                     updated_symbols.add(symbol)
#
#         if updated_symbols:
#             await self.update_positions_with_new_mark_prices(updated_symbols)
#
#     async def update_positions_with_new_mark_prices(self, updated_symbols):
#         account_data = self.last_sent_data.get('ACCOUNT_UPDATE', {})
#         positions = account_data.get('positions', [])
#         balance_data = account_data.get('balance', {})
#
#         total_unrealized_profit = Decimal('0')
#
#         for position in positions:
#             if position['symbol'] in updated_symbols:
#                 old_mark_price = Decimal(position['markPrice'])
#                 new_mark_price = Decimal(self.mark_prices[position['symbol']])
#                 position_amt = Decimal(position['positionAmt'])
#                 entry_price = Decimal(position['entryPrice'])
#
#                 # Update markPrice
#                 position['markPrice'] = str(new_mark_price)
#
#                 # Recalculate unrealizedProfit
#                 old_unrealized_profit = Decimal(position['unrealizedProfit'])
#                 new_unrealized_profit = position_amt * (new_mark_price - entry_price)
#                 position['unrealizedProfit'] = str(new_unrealized_profit)
#
#                 # Update profit_percentage
#                 position['profit_percentage'] = self.calculate_profit_percentage(position)
#
#                 # Calculate the change in unrealized profit
#                 unrealized_profit_change = new_unrealized_profit - old_unrealized_profit
#                 total_unrealized_profit += unrealized_profit_change
#
#         # Update balance data
#         if balance_data:
#             balance = Decimal(balance_data['balance'])
#             cross_wallet_balance = Decimal(balance_data['crossWalletBalance'])
#             available_balance = Decimal(balance_data['availableBalance'])
#
#             balance_data['balance'] = str(balance + total_unrealized_profit)
#             balance_data['crossWalletBalance'] = str(cross_wallet_balance)
#             balance_data['availableBalance'] = str(available_balance)
#
#         # Update the last sent data
#         self.last_sent_data['ACCOUNT_UPDATE'] = {
#             'balance': balance_data,
#             'positions': positions
#         }
#
#         # Send updated account information to the client
#         await self.send(text_data=json.dumps({
#             'type': 'ACCOUNT_UPDATE',
#             'data': self.last_sent_data['ACCOUNT_UPDATE']
#         }))
#
#     async def request_account_update(self):
#         try:
#             account_info = await self.client.futures_account()
#
#             usdt_balance = next((b for b in account_info['assets'] if b['asset'] == 'USDT'), None)
#             balance_data = None
#             if usdt_balance:
#                 balance_data = {
#                     'asset': 'USDT',
#                     'balance': usdt_balance['walletBalance'],
#                     'crossWalletBalance': usdt_balance['crossWalletBalance'],
#                     'availableBalance': usdt_balance['availableBalance']
#                 }
#
#             positions = account_info['positions']
#             active_positions = [position for position in positions if float(position['positionAmt']) != 0]
#             positions_data = []
#             if active_positions:
#                 for position in active_positions:
#                     position_data = {
#                         'symbol': position['symbol'],
#                         'positionAmt': position['positionAmt'],
#                         'entryPrice': position['entryPrice'],
#                         'unrealizedProfit': position['unrealizedProfit'],
#                         'leverage': position['leverage'],
#                         'markPrice': self.mark_prices.get(position['symbol'], position.get('markPrice', '0')),
#                         'liquidationPrice': position.get('liquidationPrice', '0')  # 청산가 추가
#
#                     }
#                     position_data['profit_percentage'] = self.calculate_profit_percentage(position_data)
#                     positions_data.append(position_data)
#
#             print(balance_data)
#             print("-------------------------------------")
#             print(positions_data)
#             await self.send_if_changed('ACCOUNT_UPDATE', {
#                 'balance': balance_data,
#                 'positions': positions_data
#             })
#
#         except Exception as e:
#             print(f"Error requesting account update: {str(e)}")
#             # print(f"Account info: {account_info}")  # 디버깅을 위해 전체 응답 출력
#             await self.send(text_data=json.dumps({
#                 'type': 'error',
#                 'message': f"Error requesting account update: {str(e)}"
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
#             if position_amt > Decimal('0'):
#                 profit_percentage = ((mark_price - entry_price) / entry_price) * 100 * leverage
#             else:
#                 profit_percentage = ((entry_price - mark_price) / entry_price) * 100 * leverage
#
#             return float(profit_percentage.quantize(Decimal('0.01')))
#         except (InvalidOperation, ZeroDivisionError):
#             return 0




