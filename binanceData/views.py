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
from .analysis.IchimokuIndicator import IchimokuIndicator
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
    def get(self, request):
        try:
            print("Processing request...")
            symbol = request.GET.get('symbol', '')
            print(symbol)
            data = self.get_bitcoin_data(symbol)
            if data is None:
                return Response({"error": "Failed to fetch data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(data)
        except Exception as e:
            print(f"Error processing request for symbol {symbol}: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get_bitcoin_data(self, symbol):
        try:
            fifteen_min_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE, limit=500)
            thirty_min_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE, limit=500)
            hourly_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, limit=500)
            daily_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, limit=500)


            # end_date = datetime.now()
            # start_date_15min = end_date - timedelta(days=7)  # Last 7 days
            # start_date_30min = end_date - timedelta(days=14)  # Last 14 days
            # start_date_hourly = end_date - timedelta(days=30)  # Last 30 days
            # start_date_daily = end_date - timedelta(days=365)  # Last 365 days
            #
            # print("Fetching data...")
            #
            # fifteen_min_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE,
            #                                                         start_date_15min.strftime("%d %b %Y %H:%M:%S"),
            #                                                         end_date.strftime("%d %b %Y %H:%M:%S"))
            #
            # thirty_min_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE,
            #                                                        start_date_30min.strftime("%d %b %Y %H:%M:%S"),
            #                                                        end_date.strftime("%d %b %Y %H:%M:%S"))
            #
            # hourly_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR,
            #                                                    start_date_hourly.strftime("%d %b %Y %H:%M:%S"),
            #                                                    end_date.strftime("%d %b %Y %H:%M:%S"))
            #
            # daily_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY,
            #                                                   start_date_daily.strftime("%d %b %Y %H:%M:%S"),
            #                                                   end_date.strftime("%d %b %Y %H:%M:%S"))

            def process_candles(candles, timeframe):
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                    'quote_asset_volume', 'number_of_trades',
                                                    'taker_buy_base_asset_volume',
                                                    'taker_buy_quote_asset_volume', 'ignore'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(
                    float)

                # Calculate additional technical indicators
                for ma in [5, 10, 20, 50, 100, 200]:
                    df[f"MA{ma}"] = df["close"].rolling(window=ma).mean()

                # Bollinger Bands
                period = 20
                multiplier = 2.0
                df["MA"] = df["close"].rolling(window=period).mean()
                df["STD"] = df["close"].rolling(window=period).std()
                df["Upper"] = df["MA"] + (df["STD"] * multiplier)
                df["Lower"] = df["MA"] - (df["STD"] * multiplier)

                # MACD
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

                StochasticRSI(df)
                RSIAnalyzer(df)
                # 시간 프레임별 일목균형표 설정값 정의
                if timeframe == '15min':
                    ichimoku_settings = {'tenkan': 7, 'kijun': 22, 'senkou_b': 44, 'displacement': 22}
                elif timeframe == '30min':
                    ichimoku_settings = {'tenkan': 7, 'kijun': 22, 'senkou_b': 44, 'displacement': 22}
                elif timeframe == 'hourly':
                    ichimoku_settings = {'tenkan': 9, 'kijun': 26, 'senkou_b': 52, 'displacement': 26}
                elif timeframe == 'daily':
                    ichimoku_settings = {'tenkan': 20, 'kijun': 60, 'senkou_b': 120, 'displacement': 30}
                else:
                    # 기본 설정값
                    ichimoku_settings = {'tenkan': 9, 'kijun': 26, 'senkou_b': 52, 'displacement': 26}

                # Stochastic RSI and RSI are already calculated, so we don't need to add them here

                ichimoku = IchimokuIndicator(df, **ichimoku_settings)
                ichimoku_df = ichimoku.get_ichimoku()
                df = pd.merge(df, ichimoku_df, on='timestamp', how='left')

                import numpy as np
                df = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)
                df['timestamp'] = df['timestamp'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

                records = df.to_dict(orient='records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value) or value in [np.inf, -np.inf]:
                            record[key] = None

                return records

            return {
                '15min': process_candles(fifteen_min_candles, '15min'),
                '30min': process_candles(thirty_min_candles, '30min'),
                'hourly': process_candles(hourly_candles, 'hourly'),
                'daily': process_candles(daily_candles, 'daily')
            }

        except Exception as e:
            print(f"An error occurred: {e}")
            return None



# class BinanceLLMChartDataAPIView(BinanceAPIView):
#     def get(self, request):
#         try:
#             print("Processing request...")
#             symbol = request.GET.get('symbol', '')
#             print(symbol)
#             data = self.get_bitcoin_data(symbol)
#             if data is None:
#                 return Response({"error": "Failed to fetch data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#             return Response(data)
#         except Exception as e:
#             print(f"Error processing request for symbol {symbol}: {str(e)}")
#             return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#
#     def get_bitcoin_data(self, symbol):
#         try:
#             end_date = datetime.now()
#             start_date_hourly = end_date - timedelta(days=21)  # 약 500시간
#             start_date_daily = end_date - timedelta(days=500)  # 500일
#             print("Fetching data...")
#
#             hourly_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR,
#                                                                start_date_hourly.strftime("%d %b %Y %H:%M:%S"),
#                                                                end_date.strftime("%d %b %Y %H:%M:%S"))
#             daily_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY,
#                                                               start_date_daily.strftime("%d %b %Y %H:%M:%S"),
#                                                               end_date.strftime("%d %b %Y %H:%M:%S"))
#
#             def process_candles(candles):
#                 df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
#                                                     'quote_asset_volume', 'number_of_trades',
#                                                     'taker_buy_base_asset_volume',
#                                                     'taker_buy_quote_asset_volume', 'ignore'])
#                 df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#                 df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
#                 df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(
#                     float)
#
#                 for ma in [5, 10, 20, 24, 50, 100, 200]:
#                     df[f"MA{ma}"] = df["close"].rolling(window=ma).mean()
#                 period = 20
#                 multiplier = 2.0
#                 df["MA"] = df["close"].rolling(window=period).mean()
#                 df["STD"] = df["close"].rolling(window=period).std()
#                 df["Upper"] = df["MA"] + (df["STD"] * multiplier)
#                 df["Lower"] = df["MA"] - (df["STD"] * multiplier)
#                 StochasticRSI(df)
#                 RSIAnalyzer(df)
#                 import numpy as np
#                 # NaN 값을 None으로 변환
#                 df = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)
#
#                 # timestamp를 ISO 형식 문자열로 변환
#                 df['timestamp'] = df['timestamp'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
#
#                 # DataFrame을 딕셔너리 리스트로 변환하고 NaN 값 추가 처리
#                 records = df.to_dict(orient='records')
#                 for record in records:
#                     for key, value in record.items():
#                         if pd.isna(value) or value in [np.inf, -np.inf]:
#                             record[key] = None
#
#                 return records
#
#             return {
#                 'hourly': process_candles(hourly_candles),
#                 'daily': process_candles(daily_candles)
#             }
#         except Exception as e:
#             print(f"An error occurred: {e}")
#             return None




