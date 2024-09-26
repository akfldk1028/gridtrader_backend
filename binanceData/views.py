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
import numpy as np
import math


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


class TrendLinesAPIView(APIView):
    def get(self, request, symbol, interval):
        try:
            df = self.fetch_binance_data(symbol, interval)
            if df is None or df.empty:
                return Response({'error': 'Failed to fetch data'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            pivot_points = self.find_pivot_points(df)
            historical_extremes = self.get_historical_extremes(df)
            if historical_extremes is None:
                return Response({'error': 'Failed to get historical extremes'},
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            trend_lines = self.generate_trend_lines(df, pivot_points, historical_extremes, symbol)

            if not trend_lines:
                return Response({'error': 'Failed to generate trend lines'},
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            top_trend_lines = self.select_top_trend_lines(trend_lines, pivot_points, historical_extremes)

            # top_trend_lines를 리스트로 평탄화
            all_top_trend_lines = []
            for trend_list in top_trend_lines.values():
                all_top_trend_lines.extend(trend_list)

            # 현재 가격이 추세선에 접근하고 있는지 확인
            approaching_trend_lines = self.check_current_price_against_trend_lines(df, all_top_trend_lines, symbol)

            response_data = {
                'pivots': pivot_points,
                'trend_lines': top_trend_lines,
                'historical_extremes': historical_extremes,
                'approaching_trend_lines': approaching_trend_lines
            }

            return Response(response_data)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get_historical_extremes(self, df):
        if df is None or df.empty:
            return None

        try:
            # 전체 기간의 역사적 고점과 저점
            historical_high = df.loc[df['High'].idxmax()]
            historical_low = df.loc[df['Low'].idxmin()]

            # 최근 데이터의 역사적 고점과 저점
            recent_df = df.tail(500)  # 최근 100개 데이터 사용
            recent_high = recent_df.loc[recent_df['High'].idxmax()]
            recent_low = recent_df.loc[recent_df['Low'].idxmin()]

            result = {
                'LongTermHigh': {
                    'Index': int(historical_high.name),
                    'Date': historical_high['Open Time'].isoformat(),
                    'Price': float(historical_high['High'])
                },
                'LongTermLow': {
                    'Index': int(historical_low.name),
                    'Date': historical_low['Open Time'].isoformat(),
                    'Price': float(historical_low['Low'])
                },
                'RecentSteepHigh': {
                    'Index': int(recent_high.name),
                    'Date': recent_high['Open Time'].isoformat(),
                    'Price': float(recent_high['High'])
                },
                'RecentSteepLow': {
                    'Index': int(recent_low.name),
                    'Date': recent_low['Open Time'].isoformat(),
                    'Price': float(recent_low['Low'])
                }
            }

            return result
        except Exception as e:
            print(f"Error in get_historical_extremes: {e}")
            return None

    # def get_historical_extremes(self, df):
    #     # 전체 기간의 역사적 고점과 저점
    #     historical_high = df.loc[df['High'].idxmax()]
    #     historical_low = df.loc[df['Low'].idxmin()]
    #
    #     # 최근 데이터 기준 가장 경사가 급한 변곡점 찾기
    #     recent_df = df.tail(100)  # 최근 100개 데이터 사용
    #     pivot_points = self.find_pivot_points(recent_df, window=5)
    #     pivot_highs = [p for p in pivot_points if p['Type'] == 'High']
    #     pivot_lows = [p for p in pivot_points if p['Type'] == 'Low']
    #
    #     steepest_high = max(pivot_highs, key=lambda x: x['Price'], default=None)
    #     steepest_low = min(pivot_lows, key=lambda x: x['Price'], default=None)
    #
    #     result = {
    #         'LongTermHigh': {
    #             'Index': int(historical_high.name),
    #             'Date': historical_high['Open Time'].isoformat(),
    #             'Price': float(historical_high['High'])
    #         },
    #         'LongTermLow': {
    #             'Index': int(historical_low.name),
    #             'Date': historical_low['Open Time'].isoformat(),
    #             'Price': float(historical_low['Low'])
    #         }
    #     }
    #
    #     if steepest_high:
    #         result['RecentSteepHigh'] = {
    #             'Index': steepest_high['Index'] + len(df) - 100,  # 전체 DataFrame에서의 인덱스로 조정
    #             'Date': steepest_high['Date'],
    #             'Price': steepest_high['Price']
    #         }
    #
    #     if steepest_low:
    #         result['RecentSteepLow'] = {
    #             'Index': steepest_low['Index'] + len(df) - 100,  # 전체 DataFrame에서의 인덱스로 조정
    #             'Date': steepest_low['Date'],
    #             'Price': steepest_low['Price']
    #         }
    #
    #     return result

    def fetch_binance_data(self, symbol, interval):
        binance_api_url = "https://api.binance.com/api/v3/klines"
        limit = 500
        total_candles = 3000
        all_candles = []

        while len(all_candles) < total_candles:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            if all_candles:
                params['endTime'] = all_candles[0][0] - 1

            try:
                response = requests.get(binance_api_url, params=params)
                response.raise_for_status()
                candles = response.json()

                if not candles:
                    break

                all_candles = candles + all_candles

            except requests.RequestException as e:
                raise Exception(f"Failed to fetch data: {str(e)}")

        # 데이터프레임 변환
        columns = [
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume", "Number of Trades",
            "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
        ]
        df = pd.DataFrame(all_candles, columns=columns)

        # 데이터 타입 변환 및 UTC로 시간대 설정
        df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", utc=True)
        df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms", utc=True)

        # 시간을 9시간 앞당기기 (KST -> UTC 변환)
        df["Open Time"] = df["Open Time"] - pd.Timedelta(hours=9)
        df["Close Time"] = df["Close Time"] - pd.Timedelta(hours=9)

        for col in ["Open", "High", "Low", "Close", "Volume", "Quote Asset Volume",
                    "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df["Number of Trades"] = df["Number of Trades"].astype(int)

        return df

    def find_pivot_points(self, df, window=1):
        pivot_points = []
        for i in range(window, len(df) - window):
            if df['High'].iloc[i] == df['High'].iloc[i - window:i + window + 1].max():
                pivot_points.append({
                    'Index': i,
                    'Date': df['Open Time'].iloc[i].isoformat(),  # UTC 시간으로 이미 설정되어 있음
                    'Price': float(df['High'].iloc[i]),
                    'Type': 'High'
                })
            elif df['Low'].iloc[i] == df['Low'].iloc[i - window:i + window + 1].min():
                pivot_points.append({
                    'Index': i,
                    'Date': df['Open Time'].iloc[i].isoformat(),  # UTC 시간으로 이미 설정되어 있음
                    'Price': float(df['Low'].iloc[i]),
                    'Type': 'Low'
                })
        return pivot_points

    def generate_trend_lines(self, df, pivot_points, historical_extremes, symbol):
        trend_lines = []

        # Long-term trend lines
        for pivot in pivot_points:
            if pivot['Type'] == 'High':
                if pivot['Index'] != historical_extremes['LongTermHigh']['Index']:
                    trend_line = self.create_trend_line(df,
                                                        historical_extremes['LongTermHigh']['Index'],
                                                        historical_extremes['RecentSteepHigh']['Index'],
                                                        historical_extremes['LongTermHigh']['Price'],
                                                        historical_extremes['RecentSteepHigh']['Price'],
                                                        'High')
                    if trend_line:
                        trend_lines.append(trend_line)
            else:  # pivot['Type'] == 'Low'
                if pivot['Index'] != historical_extremes['LongTermLow']['Index']:
                    trend_line = self.create_trend_line(df,
                                                        historical_extremes['LongTermLow']['Index'],
                                                        historical_extremes['RecentSteepLow']['Index'],
                                                        historical_extremes['LongTermLow']['Price'],
                                                        historical_extremes['RecentSteepLow']['Price'],
                                                        'Low')
                    if trend_line:
                        trend_lines.append(trend_line)

        # Recent steep trend lines
        if 'RecentSteepHigh' in historical_extremes:
            for pivot in pivot_points:
                if pivot['Type'] == 'High' and pivot['Index'] != historical_extremes['RecentSteepHigh']['Index']:
                    trend_line = self.create_trend_line(df,
                                                        historical_extremes['RecentSteepHigh']['Index'],
                                                        pivot['Index'],
                                                        historical_extremes['RecentSteepHigh']['Price'],
                                                        pivot['Price'],
                                                        'High')
                    if trend_line:
                        trend_lines.append(trend_line)

        if 'RecentSteepLow' in historical_extremes:
            for pivot in pivot_points:
                if pivot['Type'] == 'Low' and pivot['Index'] != historical_extremes['RecentSteepLow']['Index']:
                    trend_line = self.create_trend_line(df,
                                                        historical_extremes['RecentSteepLow']['Index'],
                                                        pivot['Index'],
                                                        historical_extremes['RecentSteepLow']['Price'],
                                                        pivot['Price'],
                                                        'Low')
                    if trend_line:
                        trend_lines.append(trend_line)

        return trend_lines

    def create_trend_line(self, df, start_idx, end_idx, start_price, end_price, line_type):
        start_time = df['Open Time'].iloc[start_idx]
        end_time = df['Open Time'].iloc[end_idx]
        if end_time <= start_time:
            return None  # 시간 순서가 맞지 않으면 None 반환

        # 시간과 가격 차이 계산
        time_diff = max((end_time - start_time).total_seconds(), 1e-6)  # 0으로 나누는 것 방지
        price_diff = end_price - start_price

        # 기울기 계산
        slope = price_diff / time_diff

        # 절편 계산
        start_time_seconds = (start_time - df['Open Time'].iloc[0]).total_seconds()
        intercept = start_price - slope * start_time_seconds

        # 중요도 계산 (예시로 유지)
        importance = abs(slope) * np.log1p(time_diff + 1e-6)
        if math.isnan(importance) or math.isinf(importance):
            importance = 0

        return {
            'StartIndex': int(start_idx),
            'EndIndex': int(end_idx),
            'StartDate': start_time.isoformat(),
            'EndDate': end_time.isoformat(),
            'StartPrice': float(start_price),
            'EndPrice': float(end_price),
            'Slope': float(slope),
            'Intercept': float(intercept),
            'Importance': float(importance),
            'Type': line_type
        }

    def get_current_price(self, symbol):
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error fetching current price: {response.status_code} - {response.text}")
                return None
            data = response.json()
            if 'price' not in data:
                print(f"Error: 'price' not found in response: {data}")
                return None
            return float(data['price'])
        except Exception as e:
            print(f"Exception fetching current price for {symbol}: {e}")
            return None

    def find_next_trend_line(self, trend_lines, current_line, line_type):
        same_type_lines = [line for line in trend_lines if line['Type'] == line_type]
        sorted_lines = sorted(same_type_lines, key=lambda x: x['EndIndex'])

        try:
            current_index = sorted_lines.index(current_line)
            if current_index < len(sorted_lines) - 1:
                return sorted_lines[current_index + 1]
        except ValueError:
            pass

        return None
    def check_current_price_against_trend_lines(self, df, trend_lines, symbol):
        current_price = self.get_current_price(symbol)
        print(current_price)
        if current_price is None:
            return []  # 현재 가격을 가져올 수 없으면 빈 리스트 반환


        # current_time을 얻음
        current_time = pd.Timestamp.utcnow()
        start_time = df['Open Time'].iloc[0]


        time_since_start = (current_time - start_time).total_seconds()

        approaching_lines = []

        for trend_line in trend_lines:
            # 현재 시간에 해당하는 추세선 가격 계산
            price_on_line = trend_line['Slope'] * time_since_start + trend_line['Intercept']

            # 임계값 설정 (예: 추세선 가격의 0.1%)
            threshold = 0.001 * price_on_line

            # 현재 가격이 추세선 가격에 근접한지 확인
            if abs(current_price - price_on_line) <= threshold:
                approaching_lines.append({
                    'TrendLine': trend_line,
                    'CurrentPrice': current_price,
                    'PriceOnLine': price_on_line,
                    'Difference': abs(current_price - price_on_line)
                })

        return approaching_lines

    # def check_current_price_against_trend_lines(self, df, trend_lines, symbol):
    #     current_price = self.get_current_price(symbol)
    #     print(current_price)
    #     if current_price is None:
    #         return []  # 현재 가격을 가져올 수 없으면 빈 리스트 반환
    #
    #
    #     # current_time을 얻음
    #     current_time = pd.Timestamp.utcnow()
    #     start_time = df['Open Time'].iloc[0]
    #
    #
    #     time_since_start = (current_time - start_time).total_seconds()
    #
    #     approaching_lines = []
    #
    #     for trend_line in trend_lines:
    #         # 현재 시간에 해당하는 추세선 가격 계산
    #         price_on_line = trend_line['Slope'] * time_since_start + trend_line['Intercept']
    #
    #         # 임계값 설정 (예: 추세선 가격의 0.1%)
    #         threshold = 0.001 * price_on_line
    #
    #         # 현재 가격이 추세선 가격에 근접한지 확인
    #         if abs(current_price - price_on_line) <= threshold:
    #             approaching_lines.append({
    #                 'TrendLine': trend_line,
    #                 'CurrentPrice': current_price,
    #                 'PriceOnLine': price_on_line,
    #                 'Difference': abs(current_price - price_on_line)
    #             })
    #
    #     return approaching_lines

    def select_top_trend_lines(self, trend_lines, pivot_points, historical_extremes):
        def find_nearest_opposite_pivot(start_index, end_index, line_type):
            opposite_type = 'Low' if line_type == 'High' else 'High'
            opposite_pivots = [p for p in pivot_points if
                               p['Type'] == opposite_type and start_index <= p['Index'] <= end_index]

            if not opposite_pivots:
                return None
            mid_point = (start_index + end_index) / 2
            nearest_pivot = min(opposite_pivots, key=lambda p: abs(p['Index'] - mid_point))
            return nearest_pivot

        def group_similar_slopes(lines, slope_tolerance=0.0001):
            if not lines:
                return []

            sorted_lines = sorted(lines, key=lambda x: x['Slope'])
            groups = [[sorted_lines[0]]]

            for line in sorted_lines[1:]:
                if line['Slope'] - groups[-1][0]['Slope'] <= slope_tolerance:
                    groups[-1].append(line)
                else:
                    groups.append([line])

            return [group[0] for group in groups]  # 각 그룹의 첫 번째 라인만 반환

        def process_trend_lines(lines, reverse_order=False, top_n=1):
            for line in lines:
                opposite_pivot = find_nearest_opposite_pivot(line['StartIndex'], line['EndIndex'], line['Type'])
                if opposite_pivot:
                    line['PivotDifference'] = abs(line['EndPrice'] - opposite_pivot['Price'])
                else:
                    line['PivotDifference'] = 0

            # 최종적으로 PivotDifference로 정렬
            return sorted(lines, key=lambda x: x['PivotDifference'], reverse=reverse_order)[:top_n]

        # 추세선 분류
        recent_steep_high = [line for line in trend_lines if
                             line['Type'] == 'High' and line['StartIndex'] == historical_extremes['RecentSteepHigh'][
                                 'Index']]
        recent_steep_low = [line for line in trend_lines if
                            line['Type'] == 'Low' and line['StartIndex'] == historical_extremes['RecentSteepLow'][
                                'Index']]
        long_term_high = [line for line in trend_lines if
                          line['Type'] == 'High' and line['StartIndex'] == historical_extremes['LongTermHigh']['Index']]
        long_term_low = [line for line in trend_lines if
                         line['Type'] == 'Low' and line['StartIndex'] == historical_extremes['LongTermLow']['Index']]
        # 각 분류별로 상위 10개 선택
        top_recent_steep_high = process_trend_lines(recent_steep_high, reverse_order=False, top_n=5)  # 경사도 작은 것부터
        top_recent_steep_low = process_trend_lines(recent_steep_low, reverse_order=False, top_n=5)  # 경사도 작은 것부터
        top_long_term_high = process_trend_lines(long_term_high, reverse_order=True, top_n=1)  # 경사도 큰 것부터
        top_long_term_low = process_trend_lines(long_term_low, reverse_order=True, top_n=1)

        return {
            'RecentSteepHigh': top_recent_steep_high,
            'RecentSteepLow': top_recent_steep_low,
            'LongTermHigh': top_long_term_high,
            'LongTermLow': top_long_term_low
        }

    # def select_top_trend_lines(self, trend_lines, top_n=100):
    #     # NaN이나 무한대 값 필터링
    #     valid_lines = [line for line in trend_lines if
    #                    not (math.isnan(line['Importance']) or math.isinf(line['Importance']))]
    #     sorted_lines = sorted(valid_lines, key=lambda x: x['Importance'], reverse=True)
    #     return sorted_lines[:top_n]

    def to_serializable(self, obj):
        if isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


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

    def get_extended_kline_data(self, symbol, interval, total_candles=2000):
        binance_api_url = "https://api.binance.com/api/v3/klines"
        limit = 500  # Binance API limit per request
        all_candles = []

        while len(all_candles) < total_candles:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            if all_candles:
                params['endTime'] = all_candles[0][0] - 1

            try:
                response = requests.get(binance_api_url, params=params, timeout=10)
                response.raise_for_status()
                candles = response.json()
                if not candles:
                    break
                all_candles = candles + all_candles
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {symbol}, {interval}: {e}")
                break

        return all_candles[:total_candles]

    def get_bitcoin_data(self, symbol):
        try:
            fifteen_min_candles = self.get_extended_kline_data(symbol, '15m')
            thirty_min_candles = self.get_extended_kline_data(symbol, '30m')
            hourly_candles = self.get_extended_kline_data(symbol, '1h')
            daily_candles = self.get_extended_kline_data(symbol, '6h')

            # fifteen_min_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE, limit=500)
            # thirty_min_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE, limit=500)
            # hourly_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, limit=500)
            # daily_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, limit=500)

            # end_date = datetime.now()
            # start_date_15min = end_date - timedelta(days=7)  # Last 7 days
            # start_date_30min = end_date - timedelta(days=15)  # Last 14 days
            # start_date_hourly = end_date - timedelta(days=30)  # Last 30 days
            # start_date_daily = end_date - timedelta(days=730)  # Last 365 days
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
                elif timeframe == '6hourly':
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
                'daily': process_candles(daily_candles, '6hourly')
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
