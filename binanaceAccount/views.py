from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import BinanceOrder, BinanceSymbolSettings
from binance.client import Client
from binance.exceptions import BinanceAPIException
from decimal import Decimal
from binance.client import Client
from django.conf import settings
from .serializers import BinanceOrderSerializer, BinanceSymbolSettingsSerializer
from django.utils import timezone
from datetime import timedelta
from decimal import Decimal, InvalidOperation
import time


class BinanceAPIView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Client(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)

    def sync_server_time(self):
        try:
            server_time = self.client.get_server_time()
            self.client.timestamp_offset = server_time['serverTime'] - int(time.time() * 1000)
            print(f"Synced server time. Offset: {self.client.timestamp_offset}ms")
        except BinanceAPIException as e:
            print(f"Error syncing server time: {e}")
            raise

class FuturesBalanceView(BinanceAPIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_server_time()
    def get(self, request):
        try:
            futures_balances = self.client.futures_account_balance()
            futures_usdt_balance = next((item for item in futures_balances if item["asset"] == "USDT"), None)
            return Response(futures_usdt_balance)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class ServerTimeView(BinanceAPIView):


    def get(self, request):
        try:
            server_time = self.client.get_server_time()
            return Response(server_time)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FuturesPositionView(BinanceAPIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_server_time()

    def get_positions(self, symbol=None):
        params = {}
        if symbol:
            params['symbol'] = symbol

        # 타임스탬프를 직접 추가
        params['timestamp'] = int(time.time() * 1000 + self.client.timestamp_offset)

        return self.client.futures_position_information(**params)

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

            return profit_percentage.quantize(Decimal('0.01'))
        except (InvalidOperation, ZeroDivisionError):
            return Decimal('0')

    def get(self, request):
        print(f"Received request: {request.GET}")
        try:
            # symbols = request.GET.get('symbols', '')
            # symbols = [symbol.strip().upper() for symbol in symbols.split(',') if symbol.strip()]
            symbol = request.GET.get('symbols', '')
            symbols = symbol.strip().upper() if symbol else ''

            # print(f"Processing symbols: {symbols}")

            all_positions = self.get_positions(symbols)
            # print(f"Received positions from Binance: {all_positions}")

            if symbols:
                filtered_positions = [pos for pos in all_positions if pos["symbol"] in symbols]
            else:
                filtered_positions = [pos for pos in all_positions if float(pos["positionAmt"]) != 0]

            for pos in filtered_positions:
                pos['profit_percentage'] = float(self.calculate_profit_percentage(pos))

            print(f"Returning filtered positions: {filtered_positions}")
            return Response(filtered_positions)
        except BinanceAPIException as e:
            print(f"BinanceAPIException: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return Response({'error': 'An unexpected error occurred'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FuturesAccountInfoView(BinanceAPIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_server_time()
    def get(self, request):
        try:
            futures_account_info = self.client.futures_account()
            return Response(futures_account_info)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


# class AccountInfoView(BinanceAPIView):
#     def get(self, request):
#         try:
#             # 현물 계정 정보 가져오기
#             spot_account_info = self.client.get_account()
#
#             # 선물 계정 정보 가져오기
#             futures_account_info = self.client.futures_account()
#
#             # 현물 계정 생성 또는 업데이트
#             spot_account, _ = BinanceAccount.objects.update_or_create(
#                 account_type='SPOT',
#                 defaults={
#                     'can_trade': spot_account_info['canTrade'],
#                     'can_withdraw': spot_account_info['canWithdraw'],
#                     'can_deposit': spot_account_info['canDeposit'],
#                     'update_time': spot_account_info['updateTime'],
#                     'maker_commission': spot_account_info['makerCommission'],
#                     'taker_commission': spot_account_info['takerCommission'],
#                 }
#             )
#
#             # 0보다 큰 잔고만 처리
#             spot_balances = [
#                 balance for balance in spot_account_info['balances']
#                 if Decimal(balance['free']) > 0 or Decimal(balance['locked']) > 0
#             ]
#
#             # 현물 잔고 저장
#             for balance in spot_balances:
#                 spot_balance, _ = SpotBalance.objects.update_or_create(
#                     asset=balance['asset'],
#                     defaults={
#                         'free': Decimal(balance['free']),
#                         'locked': Decimal(balance['locked']),
#                     }
#                 )
#                 spot_account.spotbalance = spot_balance
#                 spot_account.save()
#
#             # 선물 계정 생성 또는 업데이트
#             futures_account, _ = BinanceFuturesAccount.objects.update_or_create(
#                 account_type='FUTURES',
#                 defaults={
#                     'total_wallet_balance': Decimal(futures_account_info['totalWalletBalance']),
#                     'total_unrealized_profit': Decimal(futures_account_info['totalUnrealizedProfit']),
#                     'total_margin_balance': Decimal(futures_account_info['totalMarginBalance']),
#                     'available_balance': Decimal(futures_account_info['availableBalance']),
#                     'max_withdraw_amount': Decimal(futures_account_info['maxWithdrawAmount'])
#                 }
#             )
#
#             futures_balances = self.client.futures_account_balance()
#             futures_usdt_balance = next((item for item in futures_balances if item["asset"] == "USDT"), None)
#
#             if futures_usdt_balance:
#                 future_balance, _ = FutureBalance.objects.update_or_create(
#                     asset=futures_usdt_balance['asset'],
#                     defaults={
#                         'balance': Decimal(futures_usdt_balance['balance']),
#                         'cross_wallet_balance': Decimal(futures_usdt_balance.get('crossWalletBalance', '0')),
#                         'cross_un_pnl': Decimal(futures_usdt_balance.get('crossUnPnl', '0')),
#                         'available_balance': Decimal(futures_usdt_balance.get('availableBalance', '0')),
#                         'max_withdraw_amount': Decimal(futures_usdt_balance.get('maxWithdrawAmount', '0')),
#                         'margin_available': futures_usdt_balance.get('marginAvailable', False),
#                         'update_time': futures_usdt_balance.get('updateTime', None),
#                     }
#                 )
#                 futures_account.futurebalance = future_balance
#                 futures_account.save()
#
#             # 선물 포지션 정보 처리
#             symbols = request.GET.get('symbols', '').split(',')
#             symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
#
#             all_positions = self.client.futures_position_information()
#             filtered_positions = [
#                 pos for pos in all_positions
#                 if (not symbols or pos["symbol"] in symbols) and float(pos["positionAmt"]) > 0
#             ]
#
#             for position in filtered_positions:
#                 position_obj, _ = BinanceFuturePosition.objects.update_or_create(
#                     symbol=position['symbol'],
#                     defaults={
#                         'position_amt': Decimal(position['positionAmt']),
#                         'entry_price': Decimal(position['entryPrice']),
#                         'mark_price': Decimal(position['markPrice']),
#                         'un_realized_profit': Decimal(position['unRealizedProfit']),
#                         'leverage': int(position['leverage']),
#                         'profit_percentage': Decimal(position['unRealizedProfit']) / (
#                                     Decimal(position['entryPrice']) * abs(
#                                 Decimal(position['positionAmt']))) * 100 if Decimal(position['positionAmt']) != 0 else 0
#                     }
#                 )
#                 if futures_account.futurebalance:
#                     futures_account.futurebalance.positions.add(position_obj)
#
#             # 직렬화 및 응답
#             spot_account_serializer = BinanceAccountSerializer(spot_account)
#             futures_account_serializer = BinanceFuturesAccountSerializer(futures_account)
#
#             response_data = {
#                 'spot_account': spot_account_serializer.data,
#                 'futures_account': futures_account_serializer.data,
#             }
#
#             return Response(response_data)
#         except Exception as e:
#             return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class SpotAccountInfoView(BinanceAPIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_server_time()
    def get(self, request):
        try:
            spot_account_info = self.client.get_account()

            # 현물 계정 생성 또는 업데이트
            # spot_account, _ = BinanceAccount.objects.update_or_create(
            #     account_type='SPOT',
            #     defaults={
            #         'can_trade': spot_account_info['canTrade'],
            #         'can_withdraw': spot_account_info['canWithdraw'],
            #         'can_deposit': spot_account_info['canDeposit'],
            #         'update_time': spot_account_info['updateTime'],
            #         'maker_commission': spot_account_info['makerCommission'],
            #         'taker_commission': spot_account_info['takerCommission'],
            #     }
            # )
            spot_balances = [
                balance for balance in spot_account_info['balances']
                if Decimal(balance['free']) > 0 or Decimal(balance['locked']) > 0
            ]
            return Response(spot_balances)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)





class SpotBalanceView(BinanceAPIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_server_time()
    def get(self, request):
        try:
            spot_account_info = self.client.get_account()
            spot_balances = [
                balance for balance in spot_account_info['balances']
                if Decimal(balance['free']) > 0 or Decimal(balance['locked']) > 0
            ]

            return Response(spot_balances)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)









class PositionsView(BinanceAPIView):
    #TODO 각각 GET 할때 UPDATE
    pass


class LeverageView(BinanceAPIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_server_time()

    def post(self, request):
        try:
            symbols = request.data.get('symbols', [])
            leverage = int(request.data.get('leverage', 10))
            margin_type = request.data.get('margin_type', 'CROSS').upper()

            if margin_type not in ['CROSS', 'ISOLATED']:
                return Response({'error': 'Invalid margin type. Must be CROSS or ISOLATED'},
                                status=status.HTTP_400_BAD_REQUEST)

            results = []

            for symbol in symbols:
                try:
                    # Set margin type
                    self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)

                    # Set leverage
                    leverage_result = self.client.futures_change_leverage(symbol=symbol, leverage=leverage)

                    results.append({
                        'symbol': symbol,
                        'status': 'success',
                        'margin_type': margin_type,
                        'leverage_result': leverage_result
                    })

                    BinanceSymbolSettings.objects.update_or_create(
                        symbol=symbol,
                        defaults={
                            'leverage': leverage,
                            'margin_type': margin_type,
                            'account': ''
                        }
                    )
                except BinanceAPIException as e:
                    results.append({'symbol': symbol, 'status': 'error', 'message': str(e)})

            return Response(results)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class OpenOrderView(BinanceAPIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_server_time()

    def post(self, request):
        try:
            symbol = request.data.get('symbol')
            side = request.data.get('side')  # 'BUY' 또는 'SELL'
            quantity = float(request.data.get('quantity'))

            # 시장가 주문 생성
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )

            order_obj = BinanceOrder.objects.create(
                symbol=order['symbol'],
                order_id=order['orderId'],
                side=order['side'],
                type=order['type'],
                quantity=Decimal(order['origQty']),
                reduce_only=False,  # 새로운 포지션이므로 reduceOnly는 항상 False
                price=Decimal(order.get('avgPrice', '0')) if order.get('avgPrice', '0') != '0' else None,  # 체결된 가격 사용
                status=order['status'],
            )

            serializer = BinanceOrderSerializer(order_obj)
            return Response(serializer.data)
        except BinanceAPIException as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class CloseOrderView(BinanceAPIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_server_time()
    def post(self, request):
        try:
            symbol = request.data.get('symbol')
            side = request.data.get('side')  # 'SELL' 또는 'BUY'
            quantity = float(request.data.get('quantity'))

            # 'SHORT' 또는 'LONG' 대신, 'SELL' 또는 'BUY'를 사용해야 합니다.
            if side not in ['SELL', 'BUY']:
                return Response({'error': 'Invalid side. Use "SELL" for closing long positions and "BUY" for closing short positions.'}, status=status.HTTP_400_BAD_REQUEST)

            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity,
                reduceOnly=True,  # 포지션을 줄이는 주문임을 명시합니다.
            )
            print(order)
            # {'orderId': 58582697211, 'symbol': 'BNBUSDT', 'status': 'NEW', 'clientOrderId': 'YdPIbwM8dlVxmXdH0VGelJ',
            #  'price': '0.000', 'avgPrice': '0.00', 'origQty': '0.03', 'executedQty': '0.00', 'cumQty': '0.00',
            #  'cumQuote': '0.00000', 'tim
            #      eInForce': 'GTC', 'type': 'MARKET', 'reduceOnly': True, 'closePosition': False, 'side': 'BUY
            #  ', 'positionSide': 'BOTH', 'stopPrice': '0.000', 'workingType': 'CONTRACT_PRICE', 'priceProtect
            #  ': False, 'origType': 'MARKET', 'priceMatch': 'NONE', 'selfTradePreventionMode': 'NONE', 'goodTillDate
            #  ': 0, 'updateTime': 1724401088532}

            order_obj = BinanceOrder.objects.create(
                symbol=order['symbol'],
                order_id=order['orderId'],
                side=order['side'],
                type=order['type'],
                quantity=Decimal(order['origQty']),
                reduce_only=True,
                price=Decimal(order['avgPrice']),  # 체결된 평균 가격을 사용합니다.
                status=order['status'],
            )

            serializer = BinanceOrderSerializer(order_obj)
            return Response(serializer.data)
        except BinanceAPIException as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)



class BinanceFutureProfitView(BinanceAPIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_server_time()
    def get(self, request):
        try:
            end_time = int(timezone.now().timestamp() * 1000)
            start_time = int((timezone.now() - timedelta(hours=1)).timestamp() * 1000)

            income_history = self.client.futures_income_history(startTime=start_time, endTime=end_time)

            hourly_profit = sum(Decimal(income['income']) for income in income_history)

            futures_account_info = self.client.futures_account()
            current_balance = Decimal(futures_account_info['totalWalletBalance'])

            # profit_obj = BinanceFutureProfit.objects.create(
            #     timestamp=timezone.now(),
            #     hourly_profit=hourly_profit,
            #     current_balance=current_balance
            # )

            # response_data = {
            #     'timestamp': profit_obj.timestamp,
            #     'hourly_profit': float(profit_obj.hourly_profit),
            #     'current_balance': float(profit_obj.current_balance)
            # }

            # return Response(response_data)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
