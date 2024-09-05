from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
import requests
import pandas as pd
from binance.exceptions import BinanceAPIException
from decimal import Decimal
from binance.client import Client
from django.conf import settings
import time
from datetime import datetime, timedelta
from .analysis.StochasticRSI import StochasticRSI
from .analysis.rsi import RSIAnalyzer
class BinanceChartDataAPIView(APIView):
    @method_decorator(cache_page(60 * 5))  # Cache for 5 minutes
    def get(self, request, symbol, interval):
        binance_api_url = "https://api.binance.com/api/v3/klines"
        limit = 500  # Binance API limit per request
        total_candles = 3000  # Total number of candles we want to fetch

        all_candles = []

        while len(all_candles) < total_candles:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            if all_candles:
                # If we already have some candles, use the oldest one's timestamp as endTime
                params['endTime'] = all_candles[0][0] - 1

            try:
                response = requests.get(binance_api_url, params=params)
                response.raise_for_status()
                candles = response.json()

                if not candles:
                    break

                all_candles = candles + all_candles  # Prepend new candles

            except requests.RequestException as e:
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Trim excess candles
        all_candles = all_candles[:total_candles]

        # Convert to DataFrame
        columns = [
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume", "Number of Trades",
            "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
        ]
        df = pd.DataFrame(all_candles, columns=columns)

        # Convert data types
        df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
        df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume", "Quote Asset Volume",
                    "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume"]:
            df[col] = pd.to_numeric(df[col])
        df["Number of Trades"] = df["Number of Trades"].astype(int)

        # Generate JSON response
        response_data = df.to_dict('records')
        return Response(response_data)


class BinanceAPIView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Client(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
        self.sync_server_time()

    def sync_server_time(self):
        try:
            server_time = self.client.get_server_time()
            self.client.timestamp_offset = server_time['serverTime'] - int(time.time() * 1000)
            print(f"Synced server time. Offset: {self.client.timestamp_offset}ms")
        except BinanceAPIException as e:
            print(f"Error syncing server time: {e}")
            raise

class BinanceLLMChartDataAPIView(BinanceAPIView):
    def get(self, request, symbol):
        try:
            data = self.get_bitcoin_data(symbol)
            if data is None:
                return Response({"error": "Failed to fetch data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(data)
        except Exception as e:
            print(f"Error processing request for symbol {symbol}: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get_bitcoin_data(self, symbol):
        try:
            end_date = datetime.now()
            start_date_hourly = end_date - timedelta(days=21)  # 약 500시간
            start_date_daily = end_date - timedelta(days=500)  # 500일

            # 1시간 및 1일 간격의 데이터 가져오기
            hourly_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, start_date_hourly.strftime("%d %b %Y %H:%M:%S"), end_date.strftime("%d %b %Y %H:%M:%S"))
            daily_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start_date_daily.strftime("%d %b %Y %H:%M:%S"), end_date.strftime("%d %b %Y %H:%M:%S"))

            def process_candles(candles):
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                                    'taker_buy_quote_asset_volume', 'ignore'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

                for ma in [5, 10, 20, 24, 50, 100, 200]:
                    df[f"MA{ma}"] = df["close"].rolling(window=ma).mean()
                period = 20
                multiplier = 2.0
                df["MA"] = df["close"].rolling(window=period).mean()
                df["STD"] = df["close"].rolling(window=period).std()
                df["Upper"] = df["MA"] + (df["STD"] * multiplier)
                df["Lower"] = df["MA"] - (df["STD"] * multiplier)
                StochasticRSI(df)
                RSIAnalyzer(df)
                return df.to_dict(orient='records')

            return {
                'hourly': process_candles(hourly_candles),
                'daily': process_candles(daily_candles)
            }
        except Exception as e:
            print(f"An error occurred: {e}")
            return None




# class BinanceChartDataAPIView(APIView):
#     @method_decorator(cache_page(60 * 5))  # Cache for 5 minutes
#     def get(self, request, symbol, interval):
#         binance_api_url = "https://api.binance.com/api/v3/klines"
#         params = {
#             'symbol': symbol,
#             'interval': interval,
#             'limit': 500  # Binance 최대 제한
#         }
#         # end_time = int(time.time() * 1000)  # 현재 시간을 밀리초로 계산
#         # start_time = end_time - 10 * 24 * 60 * 60 * 1000  # 10일 전 시간을 밀리초로 계산
#         # params = {
#         #     "symbol": symbol,
#         #     "interval": "3m",  # 5분 간격으로 변경
#         #     "startTime": start_time,
#         #     "endTime": end_time,
#         # }
#         try:
#             response = requests.get(binance_api_url, params=params)
#             response.raise_for_status()  # Raises an HTTPError for bad responses
#             data = response.json()
#
#             # DataFrame으로 변환
#             columns = [
#                 "Open Time", "Open", "High", "Low", "Close", "Volume",
#                 "Close Time", "Quote Asset Volume", "Number of Trades",
#                 "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
#             ]
#             df = pd.DataFrame(data, columns=columns)
#
#             # 데이터 타입 변환
#             df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
#             df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
#             for col in ["Open", "High", "Low", "Close", "Volume", "Quote Asset Volume",
#                         "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume"]:
#                 df[col] = pd.to_numeric(df[col])
#             df["Number of Trades"] = df["Number of Trades"].astype(int)
#
#             # JSON 응답 생성
#             response_data = df.to_dict('records')
#             return Response(response_data)
#
#         except requests.RequestException as e:
#             return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)