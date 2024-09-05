from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import BinanceOrder, BinanceSymbolSettings,DailyBalance
from binance.client import Client
from binance.exceptions import BinanceAPIException
from decimal import Decimal
from binance.client import Client
from django.conf import settings
from .serializers import BinanceOrderSerializer, BinanceSymbolSettingsSerializer, DailyBalanceSerializer
from django.utils import timezone
from datetime import timedelta
from decimal import Decimal, InvalidOperation
import time
from typing import List, Dict, Any
import json
from datetime import date
from collections import defaultdict
import pytz

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

    def get_positions(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        params = {}
        if symbols:
            params['symbol'] = ','.join(symbols)

        return self.client.futures_position_information(**params)

    def calculate_profit_percentage(self, position: Dict[str, str]) -> Decimal:
        try:
            position_amt = Decimal(position.get('positionAmt', '0'))
            entry_price = Decimal(position.get('entryPrice', '0'))
            mark_price = Decimal(position.get('markPrice', '0'))
            leverage = Decimal(position.get('leverage', '1'))

            if position_amt == Decimal('0') or entry_price == Decimal('0'):
                return Decimal('0')

            if position_amt > Decimal('0'):  # Long position
                profit_percentage = ((mark_price - entry_price) / entry_price) * 100 * leverage
            else:  # Short position
                profit_percentage = ((entry_price - mark_price) / entry_price) * 100 * leverage

            return profit_percentage.quantize(Decimal('0.01'))
        except (InvalidOperation, ZeroDivisionError):
            print(f"Error calculating profit percentage for position: {position}")
            return Decimal('0')

    def get(self, request):
        print(f"Received request: {request.GET}")
        try:
            symbols_param = request.GET.get('symbols', '')
            symbols = [s.strip().upper() for s in symbols_param.split(',')] if symbols_param else []

            print(f"Processing symbols: {symbols}")

            all_positions = self.get_positions(symbols)
            print(f"Received {len(all_positions)} positions from Binance")

            filtered_positions = [
                pos for pos in all_positions
                if (not symbols or pos["symbol"] in symbols) and float(pos["positionAmt"]) != 0
            ]

            for pos in filtered_positions:
                pos['profit_percentage'] = float(self.calculate_profit_percentage(pos))

            print(f"Returning {len(filtered_positions)} filtered positions")
            return Response(filtered_positions)

        except BinanceAPIException as e:
            print(f"BinanceAPIException: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(f"Unexpected error in FuturesPositionView: {str(e)}")
            return Response({'error': 'An unexpected error occurred. Please try again later.'},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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

# class DailyBalanceView(BinanceAPIView):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.sync_server_time()

class DailyBalanceView(BinanceAPIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_server_time()

    def get_balance(self, balance_data):
        balance_dict = json.loads(balance_data) if isinstance(balance_data, str) else balance_data
        return float(balance_dict.get('balance', 0))

    def calculate_profit_rate(self, start_balance, end_balance):
        if start_balance == 0:
            return 0
        return ((end_balance - start_balance) / start_balance) * 100

    def get_latest_balance(self):
        try:
            futures_balances = self.client.futures_account_balance()
            futures_usdt_balance = next((item for item in futures_balances if item["asset"] == "USDT"), None)
            return float(futures_usdt_balance['balance'])
        except Exception as e:
            print(f"Error fetching latest balance: {str(e)}")
            return None

    def get_daily_profits(self, days=30):
        kst = pytz.timezone('Asia/Seoul')
        end_date = timezone.now().astimezone(kst).date()
        start_date = end_date - timedelta(days=days)

        balances = DailyBalance.objects.filter(created_at__date__gte=start_date).order_by('created_at')

        daily_data = {}
        for balance in balances:
            date = balance.created_at.astimezone(kst).date()
            if date not in daily_data or balance.created_at.astimezone(kst) < daily_data[date]['timestamp']:
                daily_data[date] = {
                    'timestamp': balance.created_at.astimezone(kst),
                    'balance': self.get_balance(balance.futures_balance)
                }

        daily_profits = []
        sorted_dates = sorted(daily_data.keys())

        for date in sorted_dates:
            daily_profits.append({
                'date': date.strftime('%Y-%m-%d'),
                'timestamp': daily_data[date]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'balance': daily_data[date]['balance'],
            })

        return daily_profits

    def find_closest_date(self, daily_profits, target_date):
        return min(daily_profits, key=lambda x: abs(date.fromisoformat(x['date']) - target_date))

    def get_profit_summary(self, daily_profits, latest_balance):
        if not daily_profits:
            return {
                '1_day': {'profit_rate': 0, 'balance': 0},
                '7_days': {'profit_rate': 0, 'balance': 0},
                '30_days': {'profit_rate': 0, 'balance': 0},
                'total': {'profit_rate': 0, 'balance': 0}
            }

        kst = pytz.timezone('Asia/Seoul')
        today = timezone.now().astimezone(kst).date()

        # 1일 수익률 (오늘)
        today_earliest = next((p for p in daily_profits if date.fromisoformat(p['date']) == today), None)
        one_day_profit = self.calculate_profit_rate(today_earliest['balance'], latest_balance) if today_earliest else 0

        # 7일 수익률
        seven_days_ago = today - timedelta(days=7)
        seven_day_start = self.find_closest_date(daily_profits, seven_days_ago)
        seven_day_profit = self.calculate_profit_rate(seven_day_start['balance'], latest_balance)

        # 30일 수익률
        thirty_days_ago = today - timedelta(days=30)
        thirty_day_start = self.find_closest_date(daily_profits, thirty_days_ago)
        thirty_day_profit = self.calculate_profit_rate(thirty_day_start['balance'], latest_balance)

        # 총 수익률
        total_profit = self.calculate_profit_rate(daily_profits[0]['balance'], latest_balance)

        return {
            '1_day': {'profit_rate': one_day_profit, 'balance': today_earliest['balance'] if today_earliest else 0},
            '7_days': {'profit_rate': seven_day_profit, 'balance': seven_day_start['balance']},
            '30_days': {'profit_rate': thirty_day_profit, 'balance': thirty_day_start['balance']},
            'total': {'profit_rate': total_profit, 'balance': daily_profits[0]['balance']}
        }

    def get(self, request):
        try:
            daily_profits = self.get_daily_profits()
            latest_balance = self.get_latest_balance()

            if latest_balance is None:
                return Response({"error": "Unable to fetch latest balance"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

            profit_summary = self.get_profit_summary(daily_profits, latest_balance)

            # Calculate profit rates for each day
            for i in range(1, len(daily_profits)):
                daily_profits[i]['profit_rate'] = self.calculate_profit_rate(daily_profits[i - 1]['balance'],
                                                                             daily_profits[i]['balance'])

            # Add today's profit rate if not already present
            kst = pytz.timezone('Asia/Seoul')
            today = timezone.now().astimezone(kst).date()
            if daily_profits and date.fromisoformat(daily_profits[-1]['date']) < today:
                today_profit_rate = self.calculate_profit_rate(daily_profits[-1]['balance'], latest_balance)
                daily_profits.append({
                    'date': today.strftime('%Y-%m-%d'),
                    'timestamp': timezone.now().astimezone(kst).strftime('%Y-%m-%d %H:%M:%S'),
                    'balance': latest_balance,
                    'profit_rate': today_profit_rate
                })

            data = {
                'daily_profits': daily_profits,
                'profit_summary': profit_summary,
                'latest_balance': latest_balance
            }

            return Response(data)
        except DailyBalance.DoesNotExist:
            return Response({"error": "No DailyBalance found"}, status=status.HTTP_404_NOT_FOUND)