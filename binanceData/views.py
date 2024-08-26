from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
import requests
import pandas as pd


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