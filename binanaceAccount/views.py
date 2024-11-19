from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import BinanceOrder, BinanceSymbolSettings, DailyBalance
from binance.client import Client
from binance.exceptions import BinanceAPIException
from decimal import Decimal
from binance.client import Client
from django.conf import settings
from .serializers import BinanceOrderSerializer, BinanceSymbolSettingsSerializer, DailyBalanceSerializer
from django.utils import timezone
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
import time
from typing import List, Dict, Any, Set

import json
from datetime import date
from collections import defaultdict
import pytz
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
import pyupbit


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
    # TODO 각각 GET 할때 UPDATE
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
                return Response({
                    'error': 'Invalid side. Use "SELL" for closing long positions and "BUY" for closing short positions.'},
                    status=status.HTTP_400_BAD_REQUEST)

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


class InvestorType(Enum):
    YOU = "DK"
    FRIEND = "DJ"
    FRIEND2 = "SK"
    FRIEND3 = "OSW"  # FRIEND3 추가


class TransactionType(Enum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"


@dataclass
class Transaction:
    date: date
    investor: InvestorType
    type: TransactionType
    amount: float


class InvestmentTracker:
    def __init__(self):
        self.transactions: List[Transaction] = []
        self.active_investors: Set[InvestorType] = set(InvestorType)

    def add_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)
        self.transactions.sort(key=lambda x: x.date)

        if transaction.type == TransactionType.WITHDRAWAL:
            # 투자금을 모두 출금한 경우 active_investors에서 제거
            if self.get_investment_amount(transaction.investor, transaction.date) == 0:
                self.active_investors.remove(transaction.investor)

    def get_investment_amount(self, investor: InvestorType, target_date: date) -> float:
        amount = 0
        for t in self.transactions:
            if t.date <= target_date and t.investor == investor:
                if t.type == TransactionType.DEPOSIT:
                    amount += t.amount
                elif t.type == TransactionType.WITHDRAWAL:
                    amount -= t.amount
        return amount

    def get_initial_investment_amount(self, investor_type: InvestorType) -> float:
        # 해당 투자자의 DEPOSIT 타입 거래 필터링
        investor_deposits = [t for t in self.transactions
                             if t.investor == investor_type and t.type == TransactionType.DEPOSIT]
        if not investor_deposits:
            return 0.0
        # 가장 빠른 투자 날짜 찾기
        earliest_date = min(t.date for t in investor_deposits)
        # 최초 투자일의 투자 금액 합산
        initial_investment = sum(t.amount for t in investor_deposits if t.date == earliest_date)
        return initial_investment

    def get_total_investment(self, target_date: date) -> float:
        return sum(self.get_investment_amount(investor, target_date) for investor in InvestorType)

    def calculate_profit_rate(self, start_balance: float, end_balance: float, investment: float) -> float:
        if investment == 0:
            return 0
        return ((end_balance - investment) / investment) * 100


class DailyBalanceView(BinanceAPIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_server_time()
        self.investment_tracker = InvestmentTracker()
        self.initialize_investments()

    def initialize_investments(self):
        self.investment_tracker.add_transaction(
            Transaction(date(2024, 9, 5), InvestorType.YOU, TransactionType.DEPOSIT, 109.58))
        self.investment_tracker.add_transaction(
            Transaction(date(2024, 9, 5), InvestorType.FRIEND, TransactionType.DEPOSIT, 40))
        self.investment_tracker.add_transaction(
            Transaction(date(2024, 9, 29), InvestorType.FRIEND, TransactionType.DEPOSIT, 72))
        self.investment_tracker.add_transaction(
            Transaction(date(2024, 9, 29), InvestorType.YOU, TransactionType.DEPOSIT, 36))
        self.investment_tracker.add_transaction(
            Transaction(date(2024, 10, 2), InvestorType.FRIEND2, TransactionType.DEPOSIT, 490))
        self.investment_tracker.add_transaction(
            Transaction(date(2024, 10, 9), InvestorType.YOU, TransactionType.DEPOSIT, 210))
        self.investment_tracker.add_transaction(
            Transaction(date(2024, 10, 9), InvestorType.FRIEND2, TransactionType.DEPOSIT, 140))
        self.investment_tracker.add_transaction(
            Transaction(date(2024, 10, 9), InvestorType.FRIEND3, TransactionType.DEPOSIT, 350))
        self.investment_tracker.add_transaction(
            Transaction(date(2024, 10, 10), InvestorType.FRIEND, TransactionType.DEPOSIT, 36))
        self.investment_tracker.add_transaction(
            Transaction(date(2024, 10, 16), InvestorType.FRIEND3, TransactionType.WITHDRAWAL, 350))
        self.investment_tracker.add_transaction(
            Transaction(date(2024, 10, 16), InvestorType.FRIEND2, TransactionType.WITHDRAWAL, 110))
        self.investment_tracker.add_transaction(
            Transaction(date(2024, 10, 29), InvestorType.YOU, TransactionType.DEPOSIT, 190))

    def get_balance(self, balance_data):
        balance_dict = json.loads(balance_data) if isinstance(balance_data, str) else balance_data
        return float(balance_dict.get('balance', 0))

    def get_latest_balance(self):
        try:
            futures_balances = self.client.futures_account_balance()
            futures_usdt_balance = next((item for item in futures_balances if item["asset"] == "USDT"), None)
            return float(futures_usdt_balance['balance'])
        except Exception as e:
            print(f"Error fetching latest balance: {str(e)}")
            return None

    def get_daily_profits(self, days=None):  # days 파라미터를 옵션으로 변경
        kst = pytz.timezone('Asia/Seoul')
        end_date = timezone.now().astimezone(kst).date()
        if days:
            start_date = end_date - timedelta(days=days)
        else:
            # days가 None이면 전체 데이터를 가져오도록 아주 이전 날짜로 설정
            from datetime import date
            start_date = date(2024, 1, 1)  # 혹은 더 이전 날짜로 설정

        balances = DailyBalance.objects.filter(created_at__date__gte=start_date).order_by('created_at')

        daily_data = {}
        for balance in balances:
            date = balance.created_at.astimezone(kst).date()
            balance_value = self.get_balance(balance.futures_balance)
            if date not in daily_data:
                daily_data[date] = {'first': balance_value, 'last': balance_value,
                                    'timestamp': balance.created_at.astimezone(kst)}
            else:
                if balance.created_at.astimezone(kst) < daily_data[date]['timestamp']:
                    daily_data[date]['first'] = balance_value
                    daily_data[date]['timestamp'] = balance.created_at.astimezone(kst)
                daily_data[date]['last'] = balance_value

        daily_profits = []
        sorted_dates = sorted(daily_data.keys())
        latest_balance = self.get_latest_balance()

        prev_you_investment = 0
        prev_friend_investment = 0
        prev_friend2_investment = 0
        prev_friend3_investment = 0
        prev_you_balance = 0
        prev_friend_balance = 0
        prev_friend2_balance = 0
        prev_friend3_balance = 0

        for i, current_date in enumerate(sorted_dates):
            if i < len(sorted_dates) - 1:
                next_date = sorted_dates[i + 1]
                end_balance = daily_data[next_date]['first']
            else:
                end_balance = latest_balance if latest_balance is not None else daily_data[current_date]['last']

            start_balance = daily_data[current_date]['first']

            # 전체 profit_rate 계산
            total_profit_rate = self.calculate_profit_rate(start_balance, end_balance)

            # 현재 날짜의 총 투자금 계산
            you_total_investment = self.investment_tracker.get_investment_amount(InvestorType.YOU, current_date)
            friend_total_investment = self.investment_tracker.get_investment_amount(InvestorType.FRIEND, current_date)
            friend2_total_investment = self.investment_tracker.get_investment_amount(InvestorType.FRIEND2, current_date)
            friend3_total_investment = self.investment_tracker.get_investment_amount(InvestorType.FRIEND3, current_date)

            # 새로운 투자금 계산
            you_new_investment = you_total_investment - prev_you_investment
            friend_new_investment = friend_total_investment - prev_friend_investment
            friend2_new_investment = friend2_total_investment - prev_friend2_investment
            friend3_new_investment = friend3_total_investment - prev_friend3_investment

            # 이전 잔액에 새 투자금 추가
            you_balance = prev_you_balance + you_new_investment
            friend_balance = prev_friend_balance + friend_new_investment
            friend2_balance = prev_friend2_balance + friend2_new_investment
            friend3_balance = max(0, prev_friend3_balance + friend3_new_investment)  # Friend3의 잔액이 0 미만이 되지 않도록 함

            # 총 투자금 및 비율 계산 (Friend3 제외)
            total_balance = you_balance + friend_balance + friend2_balance
            you_ratio = you_balance / total_balance if total_balance > 0 else 0
            friend_ratio = friend_balance / total_balance if total_balance > 0 else 0
            friend2_ratio = friend2_balance / total_balance if total_balance > 0 else 0

            # 현재 잔액을 비율에 따라 분배 (Friend3 제외)
            you_balance = end_balance * you_ratio
            friend_balance = end_balance * friend_ratio
            friend2_balance = end_balance * friend2_ratio
            friend3_balance = 0 if friend3_total_investment <= 0 else friend3_balance  # Friend3가 완전히 출금한 경우 0으로 설정

            # 수정된 profit_rate 계산
            you_profit_rate = self.investment_tracker.calculate_profit_rate(you_total_investment, you_balance,
                                                                            you_total_investment)
            friend_profit_rate = self.investment_tracker.calculate_profit_rate(friend_total_investment, friend_balance,
                                                                               friend_total_investment)
            friend2_profit_rate = self.investment_tracker.calculate_profit_rate(friend2_total_investment,
                                                                                friend2_balance,
                                                                                friend2_total_investment)
            friend3_profit_rate = self.investment_tracker.calculate_profit_rate(friend3_total_investment,
                                                                                friend3_balance,
                                                                                friend3_total_investment)

            daily_profits.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'timestamp': daily_data[current_date]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'balance': start_balance,
                'profit_rate': total_profit_rate,
                'you': {
                    'balance': you_balance,
                    'investment': you_total_investment,
                    'profit_rate': you_profit_rate
                },
                'friend': {
                    'balance': friend_balance,
                    'investment': friend_total_investment,
                    'profit_rate': friend_profit_rate
                },
                'friend2': {
                    'balance': friend2_balance,
                    'investment': friend2_total_investment,
                    'profit_rate': friend2_profit_rate
                },
                'friend3': {
                    'balance': friend3_balance,
                    'investment': friend3_total_investment,
                    'profit_rate': friend3_profit_rate
                }
            })

            # 다음 반복을 위해 현재 값을 이전 값으로 저장
            prev_you_investment = you_total_investment
            prev_friend_investment = friend_total_investment
            prev_friend2_investment = friend2_total_investment
            prev_friend3_investment = friend3_total_investment
            prev_you_balance = you_balance
            prev_friend_balance = friend_balance
            prev_friend2_balance = friend2_balance
            prev_friend3_balance = friend3_balance

        return daily_profits, latest_balance
    # def get_daily_profits(self, days=30):
    #     kst = pytz.timezone('Asia/Seoul')
    #     end_date = timezone.now().astimezone(kst).date()
    #     start_date = end_date - timedelta(days=days)
    #
    #     balances = DailyBalance.objects.filter(created_at__date__gte=start_date).order_by('created_at')
    #
    #     daily_data = {}
    #     for balance in balances:
    #         date = balance.created_at.astimezone(kst).date()
    #         balance_value = self.get_balance(balance.futures_balance)
    #         if date not in daily_data:
    #             daily_data[date] = {'first': balance_value, 'last': balance_value,
    #                                 'timestamp': balance.created_at.astimezone(kst)}
    #         else:
    #             if balance.created_at.astimezone(kst) < daily_data[date]['timestamp']:
    #                 daily_data[date]['first'] = balance_value
    #                 daily_data[date]['timestamp'] = balance.created_at.astimezone(kst)
    #             daily_data[date]['last'] = balance_value
    #
    #     daily_profits = []
    #     sorted_dates = sorted(daily_data.keys())
    #     latest_balance = self.get_latest_balance()
    #
    #     prev_you_investment = 0
    #     prev_friend_investment = 0
    #     prev_friend2_investment = 0
    #     prev_you_balance = 0
    #     prev_friend_balance = 0
    #     prev_friend2_balance = 0
    #     prev_friend3_investment = 0
    #     prev_friend3_balance = 0
    #
    #     for i, current_date in enumerate(sorted_dates):
    #         if i < len(sorted_dates) - 1:
    #             next_date = sorted_dates[i + 1]
    #             end_balance = daily_data[next_date]['first']
    #         else:
    #             end_balance = latest_balance if latest_balance is not None else daily_data[current_date]['last']
    #
    #         start_balance = daily_data[current_date]['first']
    #
    #         # 전체 profit_rate 계산 (기존 방식 유지)
    #         total_profit_rate = self.calculate_profit_rate(start_balance, end_balance)
    #
    #         # 현재 날짜의 총 투자금 계산
    #         you_total_investment = self.investment_tracker.get_investment_amount(InvestorType.YOU, current_date)
    #         friend_total_investment = self.investment_tracker.get_investment_amount(InvestorType.FRIEND, current_date)
    #         friend2_total_investment = self.investment_tracker.get_investment_amount(InvestorType.FRIEND2, current_date)
    #         friend3_total_investment = self.investment_tracker.get_investment_amount(InvestorType.FRIEND3, current_date)
    #
    #         # 새로운 투자금 계산
    #         you_new_investment = you_total_investment - prev_you_investment
    #         friend_new_investment = friend_total_investment - prev_friend_investment
    #         friend2_new_investment = friend2_total_investment - prev_friend2_investment
    #         friend3_new_investment = friend3_total_investment - prev_friend3_investment
    #
    #         # 이전 잔액에 새 투자금 추가
    #         you_balance = prev_you_balance + you_new_investment
    #         friend_balance = prev_friend_balance + friend_new_investment
    #         friend2_balance = prev_friend2_balance + friend2_new_investment
    #         friend3_balance = prev_friend3_balance + friend3_new_investment
    #
    #         # 총 투자금 및 비율 계산
    #         total_balance = you_balance + friend_balance + friend2_balance + friend3_balance
    #         you_ratio = you_balance / total_balance if total_balance > 0 else 0
    #         friend_ratio = friend_balance / total_balance if total_balance > 0 else 0
    #         friend2_ratio = friend2_balance / total_balance if total_balance > 0 else 0
    #         friend3_ratio = friend3_balance / total_balance if total_balance > 0 else 0
    #
    #         # 현재 잔액을 비율에 따라 분배
    #         you_balance = end_balance * you_ratio
    #         friend_balance = end_balance * friend_ratio
    #         friend2_balance = end_balance * friend2_ratio
    #         friend3_balance = end_balance * friend3_ratio
    #
    #         # you_balance = start_balance * you_ratio
    #         # friend_balance = start_balance * friend_ratio
    #
    #         # 수정된 profit_rate 계산
    #         you_profit_rate = self.investment_tracker.calculate_profit_rate(you_total_investment, you_balance,
    #                                                                         you_total_investment)
    #         friend_profit_rate = self.investment_tracker.calculate_profit_rate(friend_total_investment, friend_balance,
    #                                                                            friend_total_investment)
    #         friend2_profit_rate = self.investment_tracker.calculate_profit_rate(friend2_total_investment,
    #                                                                             friend2_balance,
    #                                                                             friend2_total_investment)
    #         friend3_profit_rate = self.investment_tracker.calculate_profit_rate(friend3_total_investment,
    #                                                                             friend3_balance,
    #                                                                             friend3_total_investment)
    #
    #         daily_profits.append({
    #             'date': current_date.strftime('%Y-%m-%d'),
    #             'timestamp': daily_data[current_date]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
    #             'balance': start_balance,
    #             'profit_rate': total_profit_rate,
    #             'you': {
    #                 'balance': you_balance,
    #                 'investment': you_total_investment,
    #                 'profit_rate': you_profit_rate
    #             },
    #             'friend': {
    #                 'balance': friend_balance,
    #                 'investment': friend_total_investment,
    #                 'profit_rate': friend_profit_rate
    #             },
    #             'friend2': {
    #                 'balance': friend2_balance,
    #                 'investment': friend2_total_investment,
    #                 'profit_rate': friend2_profit_rate
    #             },
    #             'friend3': {
    #                 'balance': friend3_balance,
    #                 'investment': friend3_total_investment,
    #                 'profit_rate': friend3_profit_rate
    #             }
    #         })
    #
    #         # 다음 반복을 위해 현재 값을 이전 값으로 저장
    #         prev_you_investment = you_total_investment
    #         prev_friend_investment = friend_total_investment
    #         prev_friend2_investment = friend2_total_investment
    #         prev_you_balance = you_balance
    #         prev_friend_balance = friend_balance
    #         prev_friend2_balance = friend2_balance
    #         prev_friend3_investment = friend3_total_investment
    #         prev_friend3_balance = friend3_balance
    #
    #     return daily_profits, latest_balance

    def calculate_profit_rate(self, start_balance: float, end_balance: float) -> float:
        if start_balance == 0:
            return 0
        return ((end_balance - start_balance) / start_balance) * 100

    def get_profit_summary(self, daily_profits, latest_balance):
        if not daily_profits:
            return {
                '1_day': {'profit_rate': 0, 'balance': 0, 'you': {'balance': 0, 'profit_rate': 0},
                          'friend': {'balance': 0, 'profit_rate': 0}},
                '7_days': {'profit_rate': 0, 'balance': 0, 'you': {'balance': 0, 'profit_rate': 0},
                           'friend': {'balance': 0, 'profit_rate': 0}},
                '30_days': {'profit_rate': 0, 'balance': 0, 'you': {'balance': 0, 'profit_rate': 0},
                            'friend': {'balance': 0, 'profit_rate': 0}},
                'total': {'profit_rate': 0, 'balance': 0, 'you': {'balance': 0, 'profit_rate': 0},
                          'friend': {'balance': 0, 'profit_rate': 0}}
            }

        kst = pytz.timezone('Asia/Seoul')
        today = timezone.now().astimezone(kst).date()

        def get_period_data(days=None):
            if days is None or days == 0:
                # 'total'의 경우, 각 투자자의 초기 투자 금액을 직접 사용
                you_initial_investment = self.investment_tracker.get_initial_investment_amount(InvestorType.YOU)
                friend_initial_investment = self.investment_tracker.get_initial_investment_amount(InvestorType.FRIEND)
                friend2_initial_investment = self.investment_tracker.get_initial_investment_amount(InvestorType.FRIEND2)
                friend3_initial_investment = self.investment_tracker.get_initial_investment_amount(InvestorType.FRIEND3)

                start_balance_you = you_initial_investment
                start_balance_friend = friend_initial_investment
                start_balance_friend2 = friend2_initial_investment
                start_balance_friend3 = friend3_initial_investment

            else:
                target_date = today - timedelta(days=days)
                start_data = next(
                    (p for p in reversed(daily_profits) if date.fromisoformat(p['date']) <= target_date),
                    daily_profits[0]
                )
                start_balance_you = start_data['you']['balance']
                start_balance_friend = start_data['friend']['balance']
                start_balance_friend2 = start_data['friend2']['balance']
                start_balance_friend3 = start_data['friend3']['balance']

            end_data = daily_profits[-1]
            end_balance_you = end_data['you']['balance']
            end_balance_friend = end_data['friend']['balance']
            end_balance_friend2 = end_data['friend2']['balance']
            end_balance_friend3 = end_data['friend3']['balance']

            you_profit_rate = self.calculate_profit_rate(start_balance_you, end_balance_you)
            friend_profit_rate = self.calculate_profit_rate(start_balance_friend, end_balance_friend)
            friend2_profit_rate = self.calculate_profit_rate(start_balance_friend2, end_balance_friend2)
            friend3_profit_rate = self.calculate_profit_rate(start_balance_friend3, end_balance_friend3)

            return {
                'you': {
                    'startBalance': start_balance_you,
                    'endBalance': end_balance_you,
                    'profit_rate': you_profit_rate
                },
                'friend': {
                    'startBalance': start_balance_friend,
                    'endBalance': end_balance_friend,
                    'profit_rate': friend_profit_rate
                },
                'friend2': {
                    'startBalance': start_balance_friend2,
                    'endBalance': end_balance_friend2,
                    'profit_rate': friend2_profit_rate
                },
                'friend3': {
                    'startBalance': start_balance_friend3,
                    'endBalance': end_balance_friend3,
                    'profit_rate': friend3_profit_rate
                }
            }

        # 1일 수익률 (오늘)
        today_earliest = next((p for p in daily_profits if date.fromisoformat(p['date']) == today), None)
        one_day_profit = self.calculate_profit_rate(today_earliest['balance'], latest_balance) if today_earliest else 0
        one_day_data = get_period_data(1)

        # 7일 수익률
        seven_days_ago = today - timedelta(days=7)
        seven_day_start = self.find_closest_date(daily_profits, seven_days_ago)
        seven_day_profit = self.calculate_profit_rate(seven_day_start['balance'], latest_balance)
        seven_day_data = get_period_data(7)

        # 30일 수익률
        thirty_days_ago = today - timedelta(days=30)
        thirty_day_start = self.find_closest_date(daily_profits, thirty_days_ago)
        thirty_day_profit = self.calculate_profit_rate(thirty_day_start['balance'], latest_balance)
        thirty_day_data = get_period_data(30)

        # 총 수익률
        # 총 수익률 계산 시 최초 투자 금액만 사용
        total_initial_investment = 149.58
        total_profit = self.calculate_profit_rate(total_initial_investment, latest_balance)
        total_data = get_period_data()

        return {
            '1_day': {
                'profit_rate': one_day_profit,
                'balance': today_earliest['balance'] if today_earliest else 0,
                'you': one_day_data['you'],
                'friend': one_day_data['friend'],
                'friend2': one_day_data['friend2'],
                'friend3': one_day_data['friend3']

            },
            '7_days': {
                'profit_rate': seven_day_profit,
                'balance': seven_day_start['balance'],
                'you': seven_day_data['you'],
                'friend': seven_day_data['friend'],
                'friend2': seven_day_data['friend2'],
                'friend3': seven_day_data['friend3']

            },
            '30_days': {
                'profit_rate': thirty_day_profit,
                'balance': thirty_day_start['balance'],
                'you': thirty_day_data['you'],
                'friend': thirty_day_data['friend'],
                'friend2': thirty_day_data['friend2'],
                'friend3': thirty_day_data['friend3']

            },
            'total': {
                'profit_rate': total_profit,
                'balance': daily_profits[0]['balance'],
                'you': total_data['you'],
                'friend': total_data['friend'],
                'friend2': total_data['friend2'],
                'friend3': total_data['friend3']

            }
        }

    def find_closest_date(self, daily_profits, target_date):
        return min(daily_profits, key=lambda x: abs(date.fromisoformat(x['date']) - target_date))

    def get(self, request):
        try:
            calculation_date_str = request.query_params.get('date', timezone.now().date().isoformat())
            calculation_date = datetime.strptime(calculation_date_str, "%Y-%m-%d").date()

            daily_profits, latest_balance = self.get_daily_profits(days=None)

            if latest_balance is None:
                return Response({"error": "Unable to fetch latest balance"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

            profit_summary = self.get_profit_summary(daily_profits, latest_balance)

            data = {
                'calculation_date': calculation_date.isoformat(),
                'daily_profits': daily_profits,
                'profit_summary': profit_summary,
                'latest_balance': latest_balance,
            }

            return Response(data)
        except DailyBalance.DoesNotExist:
            return Response({"error": "No DailyBalance found"}, status=status.HTTP_404_NOT_FOUND)


# views.py


class UpbitBaseView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.upbit = pyupbit.Upbit(
            settings.UPBIT_ACCESS_KEY,
            settings.UPBIT_SECRET_KEY
        )


class BalanceView(UpbitBaseView):
    def get(self, request):
        """
        잔고 조회 API
        GET /api/v1/balance/
        """
        try:
            balances = self.upbit.get_balances()
            formatted_balances = {}

            for balance in balances:
                currency = balance['currency']
                formatted_balances[currency] = {
                    'balance': float(balance['balance']),
                    'avg_buy_price': float(balance['avg_buy_price']) if balance['avg_buy_price'] else 0,
                    'unit_currency': balance['unit_currency']
                }

            return Response({
                'status': 'success',
                'data': formatted_balances
            })

        except Exception as e:
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TradeView(UpbitBaseView):
    def post(self, request):
        """
        거래 실행 API
        POST /api/v1/trade/

        Request Body:
        {
            "action": "buy" | "sell",
            "market": "KRW-BTC",
            "is_full_trade": false  # true면 전체 거래, false면 50% 거래
        }
        """
        try:
            action = request.data.get('action')
            market = request.data.get('market', 'KRW-BTC')
            percentage = float(request.data.get('percentage', 100))
            is_full_trade = request.data.get('is_full_trade', False)

            if action not in ['buy', 'sell']:
                return Response({
                    'status': 'error',
                    'message': 'Invalid action. Must be either "buy" or "sell"'
                }, status=status.HTTP_400_BAD_REQUEST)

            # 코인의 평균매수가 조회
            coin_currency = market.split('-')[1]
            avg_buy_price = 0
            balances = self.upbit.get_balances()
            for b in balances:
                if b['currency'] == coin_currency:
                    avg_buy_price = float(b.get('avg_buy_price', 0))
                    break

            if action == 'buy':
                # 기존 buy 로직
                krw_balance = float(self.upbit.get_balance("KRW"))
                percentage = 100 if is_full_trade else 50
                amount_to_invest = krw_balance * (percentage / 100)

                if amount_to_invest < 5000:
                    return Response({
                        'status': 'error',
                        'message': 'Minimum order amount is 5000 KRW'
                    }, status=status.HTTP_400_BAD_REQUEST)

                result = self.upbit.buy_market_order(market, amount_to_invest * 0.9995)

            else:  # sell
                # 기존 sell 로직
                coin_balance = float(self.upbit.get_balance(coin_currency))
                percentage = 100 if is_full_trade else 50
                amount_to_sell = coin_balance * (percentage / 100)

                current_price = pyupbit.get_orderbook(ticker=market)['orderbook_units'][0]["ask_price"]
                if current_price * amount_to_sell < 5000:
                    return Response({
                        'status': 'error',
                        'message': 'Order amount is less than minimum requirement (5000 KRW)'
                    }, status=status.HTTP_400_BAD_REQUEST)

                result = self.upbit.sell_market_order(market, amount_to_sell)

            # 거래 후 업데이트된 평균매수가 조회
            updated_avg_buy_price = 0
            updated_balances = self.upbit.get_balances()
            for b in updated_balances:
                if b['currency'] == coin_currency:
                    updated_avg_buy_price = float(b.get('avg_buy_price', 0))
                    break

            return Response({
                'status': 'success',
                'data': {
                    'result': result,
                    'action': action,
                    'is_full_trade': is_full_trade,
                    'percentage': 100 if is_full_trade else 50,
                    'previous_avg_buy_price': avg_buy_price,  # 거래 전 평균매수가
                    'updated_avg_buy_price': updated_avg_buy_price  # 거래 후 평균매수가
                }
            })

        except Exception as e:
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class TradeView(UpbitBaseView):
#     def post(self, request):
#         """
#         거래 실행 API
#         POST /api/v1/trade/
#
#         Request Body:
#         {
#             "action": "buy" | "sell",
#             "market": "KRW-BTC",
#             "is_full_trade": false  # true면 전체 거래, false면 50% 거래
#         }
#         """
#         try:
#             action = request.data.get('action')
#             market = request.data.get('market', 'KRW-BTC')
#             percentage = float(request.data.get('percentage', 100))
#             is_full_trade = request.data.get('is_full_trade', False)
#
#             if action not in ['buy', 'sell']:
#                 return Response({
#                     'status': 'error',
#                     'message': 'Invalid action. Must be either "buy" or "sell"'
#                 }, status=status.HTTP_400_BAD_REQUEST)
#
#             if action == 'buy':
#                 # KRW 잔고 확인
#                 krw_balance = float(self.upbit.get_balance("KRW"))
#                 # is_full_trade에 따라 거래 비율 결정
#                 percentage = 100 if is_full_trade else 50
#                 amount_to_invest = krw_balance * (percentage / 100)
#
#                 if amount_to_invest < 5000:
#                     return Response({
#                         'status': 'error',
#                         'message': 'Minimum order amount is 5000 KRW'
#                     }, status=status.HTTP_400_BAD_REQUEST)
#
#                 # 수수료를 고려한 주문 실행
#                 result = self.upbit.buy_market_order(market, amount_to_invest * 0.9995)
#
#             else:  # sell
#                 # 코인 잔고 확인
#                 coin_currency = market.split('-')[1]
#                 coin_balance = float(self.upbit.get_balance(coin_currency))
#
#                 # is_full_trade에 따라 거래 비율 결정
#                 percentage = 100 if is_full_trade else 50
#                 amount_to_sell = coin_balance * (percentage / 100)
#
#                 # 최소 주문 금액 확인
#                 current_price = pyupbit.get_orderbook(ticker=market)['orderbook_units'][0]["ask_price"]
#                 if current_price * amount_to_sell < 5000:
#                     return Response({
#                         'status': 'error',
#                         'message': 'Order amount is less than minimum requirement (5000 KRW)'
#                     }, status=status.HTTP_400_BAD_REQUEST)
#
#                 result = self.upbit.sell_market_order(market, amount_to_sell)
#
#             return Response({
#                 'status': 'success',
#                 'data': {
#                     'result': result,
#                     'action': action,
#                     'is_full_trade': is_full_trade,
#                     'percentage': 100 if is_full_trade else 50
#                 }
#             })
#
#         except Exception as e:
#             return Response({
#                 'status': 'error',
#                 'message': str(e)
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# class CurrentPriceView(UpbitBaseView):
#     def get(self, request):
#         """
#         현재가 조회 API
#         GET /api/v1/price/?market=KRW-BTC
#         """
#         try:
#             market = request.query_params.get('market', 'KRW-BTC')
#             orderbook = pyupbit.get_orderbook(ticker=market)
#
#             return Response({
#                 'status': 'success',
#                 'data': {
#                     'market': market,
#                     'timestamp': orderbook['timestamp'],
#                     'ask_price': orderbook['orderbook_units'][0]["ask_price"],
#                     'bid_price': orderbook['orderbook_units'][0]["bid_price"]
#                 }
#             })
#
#         except Exception as e:
#             return Response({
#                 'status': 'error',
#                 'message': str(e)
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class CurrentPriceView(UpbitBaseView):
    def get(self, request):
        """
        현재가 조회 API
        GET /api/v1/price/?market=KRW-BTC
        """
        try:
            market = request.query_params.get('market', 'KRW-BTC')
            orderbook = pyupbit.get_orderbook(ticker=market)

            # 코인의 평균매수가 조회
            coin_currency = market.split('-')[1]
            avg_buy_price = 0
            balances = self.upbit.get_balances()
            for b in balances:
                if b['currency'] == coin_currency:
                    avg_buy_price = float(b.get('avg_buy_price', 0))
                    break

            return Response({
                'status': 'success',
                'data': {
                    'market': market,
                    'timestamp': orderbook['timestamp'],
                    'ask_price': orderbook['orderbook_units'][0]["ask_price"],
                    'bid_price': orderbook['orderbook_units'][0]["bid_price"],
                    'avg_buy_price': avg_buy_price  # 평균매수가 추가
                }
            })

        except Exception as e:
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)