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
from .analysis.UltimateRSIAnalyzer import UltimateRSIAnalyzer
from .analysis.SqueezeMomentumIndicator import SqueezeMomentumIndicator
import concurrent.futures
from .models import BinanceTradingSummary, KoreaStockData, StockData , ChinaStockData


from .analysis.IchimokuIndicator import IchimokuIndicator
import numpy as np
import math
from typing import List, Dict, Optional
import bisect
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from functools import lru_cache
from django.core.cache import caches
import json
from pandas import Timestamp
from datetime import datetime, timezone
import pandas_ta as ta
import pyupbit
from typing import Dict, List, Any
from .models import TradingRecord
import yfinance as yf
from pykrx import stock


class KRXStockDataAPIView(APIView):
    @method_decorator(cache_page(60 * 15))  # Cache for 15 minutes
    def get(self, request, symbol, start_date, end_date, interval='D'):
        krx_api_url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

        params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'locale': 'ko_KR',
            'tboxisuCd_finder_stkisu0_0': symbol,
            'isuCd': symbol,
            'strtDd': start_date,
            'endDd': end_date,
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false'
        }

        try:
            response = requests.post(krx_api_url, data=params)
            response.raise_for_status()
            data = response.json()

            if 'OutBlock_1' not in data:
                return Response({'error': 'No data available'}, status=status.HTTP_404_NOT_FOUND)

            df = pd.DataFrame(data['OutBlock_1'])

            # Convert data types
            df['TRD_DD'] = pd.to_datetime(df['TRD_DD'])
            for col in ['TDD_CLSPRC', 'TDD_OPNPRC', 'TDD_HGPRC', 'TDD_LWPRC', 'ACC_TRDVOL', 'ACC_TRDVAL']:
                df[col] = pd.to_numeric(df[col].str.replace(',', ''))

            # Rename columns
            df = df.rename(columns={
                'TRD_DD': 'Date',
                'TDD_CLSPRC': 'Close',
                'TDD_OPNPRC': 'Open',
                'TDD_HGPRC': 'High',
                'TDD_LWPRC': 'Low',
                'ACC_TRDVOL': 'Volume',
                'ACC_TRDVAL': 'Trade Value'
            })

            # Sort by date
            df = df.sort_values('Date')

            if interval.upper() == 'W':
                # Resample to weekly data
                df = df.set_index('Date')
                weekly_df = df.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum',
                    'Trade Value': 'sum'
                }).reset_index()
                df = weekly_df

            # Generate JSON response
            response_data = df.to_dict('records')
            return Response(response_data)

        except requests.RequestException as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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


class TrendLineDataRetrieveView(APIView):
    cache_name = 'default'  # TrendLinesAPIView와 동일한 캐시 사용
    cache_key_prefix = 'trend_line_data'

    def get_cache(self):
        return caches[self.cache_name]

    def get_cache_key(self, symbol, interval):
        return f"{self.cache_key_prefix}:{symbol}:{interval}"

    def get(self, request, symbol, interval):
        cache = self.get_cache()
        key = self.get_cache_key(symbol, interval)

        try:
            stored_data = cache.get(key)
            if stored_data:
                decoded_data = json.loads(stored_data)
                return Response(decoded_data, status=status.HTTP_200_OK)
            else:
                return Response({"error": "No data found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": f"Failed to retrieve data: {str(e)}"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TrendLinesAPIView(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_states = {}  # 이전 상태를 저장할 딕셔너리

    cache_name = 'default'  # 또는 원하는 캐시 이름
    cache_key_prefix = 'trend_line_data'

    def get_cache(self):
        return caches[self.cache_name]

    def get_cache_key(self, symbol, interval):
        return f"{self.cache_key_prefix}:{symbol}:{interval}"

    def get(self, request, symbol, interval):
        try:

            cache = self.get_cache()
            key = self.get_cache_key(symbol, interval)
            stored_data = cache.get(key)
            if stored_data:
                previous_data = json.loads(stored_data)
                self.previous_states = {
                    line['id']: (line.get('CurrentState', ''), None)
                    for category, lines in previous_data['trend_lines'].items()
                    for line in lines
                }

            df = self.fetch_binance_data(symbol, interval)
            if df is None or df.empty:
                return Response({'error': 'Failed to fetch data'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # 피벗 포인트 찾기 전에 최신 데이터 포함
            pivot_points = self.find_pivot_points(df)
            historical_extremes = self.get_historical_extremes(df, interval)

            if historical_extremes is None:
                return Response({'error': 'Failed to get historical extremes'},
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # 최근 고점과 저점을 피벗 포인트에 추가
            self.add_recent_extremes_to_pivots(pivot_points, historical_extremes)

            trend_lines = self.generate_trend_lines(df, pivot_points, historical_extremes, symbol)
            if not trend_lines:
                return Response({'error': 'Failed to generate trend lines'},
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            top_trend_lines = self.select_top_trend_lines(trend_lines, pivot_points, historical_extremes)
            print(f"Top trend lines: {top_trend_lines}")

            # top_trend_lines를 리스트로 평탄화
            all_top_trend_lines = []
            for category, trend_list in top_trend_lines.items():
                for line in trend_list:
                    line['id'] = f"{line['StartIndex']}_{line['EndIndex']}"
                    all_top_trend_lines.append(line)

            updated_trend_lines = self.check_current_price_against_trend_lines(df, all_top_trend_lines, symbol,
                                                                               interval)
            # top_trend_lines 업데이트
            print("2222222222222")
            for category in top_trend_lines:
                top_trend_lines[category] = [line for line in updated_trend_lines if line in top_trend_lines[category]]

            response_data = {
                'pivots': pivot_points,
                'trend_lines': top_trend_lines,  # 업데이트된 trend_lines
                'historical_extremes': historical_extremes,
                'symbol': symbol,
                'interval': interval
            }

            def process_timestamps(data):
                if isinstance(data, dict):
                    return {k: process_timestamps(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [process_timestamps(v) for v in data]
                elif isinstance(data, Timestamp):
                    return data.isoformat()
                else:
                    return data

            # 사용 예:
            response_data = process_timestamps(response_data)
            cache.set(key, json.dumps(response_data), timeout=None)

            return Response(response_data)

        except Exception as e:
            print(f"Exception occurred: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def add_recent_extremes_to_pivots(self, pivot_points, historical_extremes):
        for extreme_type in ['RecentSteepHigh', 'RecentSteepLow']:
            if extreme_type in historical_extremes:
                extreme = historical_extremes[extreme_type]
                pivot_type = 'High' if 'High' in extreme_type else 'Low'
                new_pivot = {
                    'Index': extreme['Index'],
                    'Date': extreme['Date'],
                    'Price': extreme['Price'],
                    'Type': pivot_type
                }
                if new_pivot not in pivot_points:
                    pivot_points.append(new_pivot)

        # 인덱스로 정렬
        pivot_points.sort(key=lambda x: x['Index'])

    def get_historical_extremes(self, df, interval):
        if df is None or df.empty:
            print("DataFrame is None or empty in get_historical_extremes")
            return None

        try:
            # 전체 기간의 역사적 고점과 저점
            historical_high = df.loc[df['High'].idxmax()]
            historical_low = df.loc[df['Low'].idxmin()]

            # 최근 데이터의 역사적 고점과 저점
            recent_data_count = {
                '5m': 900,  # 약 3일
                '15m': 720,  # 약 10일
                '30m': 600,  # 약 20일
                '1h': 600,  # 약 30일
                '2h': 1020,  # �� 41일
                '4h': 480,  # 약 83일
                '1d': 240,  # 약 1년
                '3d': 180,
                '1w': 120,  # 약 4년
            }

            count = recent_data_count.get(interval, 500)  # 기본값은 500
            recent_df = df.tail(count)
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

        # # 시간을 9시간 앞당기기 (KST -> UTC 변환)
        # df["Open Time"] = df["Open Time"] - pd.Timedelta(hours=9)
        # df["Close Time"] = df["Close Time"] - pd.Timedelta(hours=9)

        for col in ["Open", "High", "Low", "Close", "Volume", "Quote Asset Volume",
                    "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df["Number of Trades"] = df["Number of Trades"].astype(int)

        return df

    def find_pivot_points(self, df, window=5):
        pivot_points = []
        for i in range(window, len(df) - window):
            if df['High'].iloc[i] == df['High'].iloc[i - window:i + window + 1].max():
                pivot_points.append({
                    'Index': i,
                    'Date': df['Open Time'].iloc[i].isoformat(),
                    'Price': float(df['High'].iloc[i]),
                    'Type': 'High'
                })
            elif df['Low'].iloc[i] == df['Low'].iloc[i - window:i + window + 1].min():
                pivot_points.append({
                    'Index': i,
                    'Date': df['Open Time'].iloc[i].isoformat(),
                    'Price': float(df['Low'].iloc[i]),
                    'Type': 'Low'
                })
        return pivot_points

    def generate_trend_lines(self, df, pivot_points, historical_extremes, symbol):
        trend_lines = []

        try:
            # Long-term trend lines
            for pivot in pivot_points:
                if pivot['Type'] == 'High':
                    if pivot['Index'] != historical_extremes['LongTermHigh']['Index']:
                        trend_line = self.create_trend_line(
                            df,
                            historical_extremes['LongTermHigh']['Index'],
                            historical_extremes['RecentSteepHigh']['Index'],
                            historical_extremes['LongTermHigh']['Price'],
                            historical_extremes['RecentSteepHigh']['Price'],
                            'High',
                            pivot_points
                        )
                        if trend_line:
                            trend_lines.append(trend_line)
                else:  # pivot['Type'] == 'Low'
                    if pivot['Index'] != historical_extremes['LongTermLow']['Index']:
                        trend_line = self.create_trend_line(
                            df,
                            historical_extremes['LongTermLow']['Index'],
                            historical_extremes['RecentSteepLow']['Index'],
                            historical_extremes['LongTermLow']['Price'],
                            historical_extremes['RecentSteepLow']['Price'],
                            'Low',
                            pivot_points
                        )
                        if trend_line:
                            trend_lines.append(trend_line)

            # Recent steep trend lines
            if 'RecentSteepHigh' in historical_extremes:
                for pivot in pivot_points:
                    if pivot['Type'] == 'High' and pivot['Index'] != historical_extremes['RecentSteepHigh']['Index']:
                        trend_line = self.create_trend_line(
                            df,
                            historical_extremes['RecentSteepHigh']['Index'],
                            pivot['Index'],
                            historical_extremes['RecentSteepHigh']['Price'],
                            pivot['Price'],
                            'High',
                            pivot_points
                        )
                        if trend_line:
                            trend_lines.append(trend_line)

            if 'RecentSteepLow' in historical_extremes:
                for pivot in pivot_points:
                    if pivot['Type'] == 'Low' and pivot['Index'] != historical_extremes['RecentSteepLow']['Index']:
                        trend_line = self.create_trend_line(
                            df,
                            historical_extremes['RecentSteepLow']['Index'],
                            pivot['Index'],
                            historical_extremes['RecentSteepLow']['Price'],
                            pivot['Price'],
                            'Low',
                            pivot_points
                        )
                        if trend_line:
                            trend_lines.append(trend_line)
        except KeyError as e:
            print(f"KeyError: Missing key in historical_extremes or pivot_points: {e}")
        except TypeError as e:
            print(f"TypeError: Invalid data type encountered: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return trend_lines

    def create_trend_line(self, df, start_idx, end_idx, start_price, end_price, line_type, pivot_points):
        start_time = df['Open Time'].iloc[start_idx]
        end_time = df['Open Time'].iloc[end_idx]
        if end_time <= start_time:
            print(f"Invalid time order in create_trend_line: start_time={start_time}, end_time={end_time}")
            return None  # 시간 순서가 맞지 않으면 None 반환

        # 시간과 가격 차이 계산
        # 시간과 가격 차이 계산
        time_diff = (end_time - start_time).total_seconds()
        price_diff = end_price - start_price

        # 기울기 계산
        slope = price_diff / time_diff

        # 절편 계산
        intercept = start_price - slope * start_time.timestamp()

        opposite_type = 'Low' if line_type == 'High' else 'High'
        current_pivot = next(p for p in pivot_points if p['Index'] == end_idx and p['Type'] == line_type)
        prev_pivot = next((p for p in reversed(pivot_points) if p['Type'] == opposite_type and p['Index'] < end_idx),
                          None)
        next_pivot = next((p for p in pivot_points if p['Type'] == opposite_type and p['Index'] > end_idx), None)

        def calculate_slope(p1, p2):
            if p1 and p2:
                return (p2['Price'] - p1['Price']) / (p2['Index'] - p1['Index'])
            return None

        prev_slope = calculate_slope(prev_pivot, current_pivot)
        next_slope = calculate_slope(current_pivot, next_pivot)

        def calculate_price_diff(p1, p2):
            if p1 and p2:
                return abs(p2['Price'] - p1['Price'])
            return None

        prev_price_diff = calculate_price_diff(prev_pivot, current_pivot)
        next_price_diff = calculate_price_diff(current_pivot, next_pivot)
        # 가격 차이의 상대적 크기 계산
        max_price = max(start_price, end_price)
        min_price = min(start_price, end_price)
        price_range = max_price - min_price

        prev_relative_diff = prev_price_diff / price_range if prev_price_diff and price_range else 0
        next_relative_diff = next_price_diff / price_range if next_price_diff and price_range else 0

        return {
            'StartIndex': int(start_idx),
            'EndIndex': int(end_idx),
            'StartDate': start_time.isoformat(),
            'EndDate': end_time.isoformat(),
            'StartPrice': float(start_price),
            'EndPrice': float(end_price),
            'Slope': float(slope),
            'Intercept': float(intercept),
            'Type': line_type,
            'CurrentPivot': {
                'Index': int(current_pivot['Index']),
                'Date': current_pivot['Date'],
                'Price': float(current_pivot['Price'])
            },
            'PrevPivot': {
                'Index': int(prev_pivot['Index']) if prev_pivot else None,
                'Date': prev_pivot['Date'] if prev_pivot else None,
                'Price': float(prev_pivot['Price']) if prev_pivot else None,
                'Slope': float(prev_slope) if prev_slope else None,
                'PriceDiff': float(prev_price_diff) if prev_price_diff else None,
                'RelativeDiff': float(prev_relative_diff)
            },
            'NextPivot': {
                'Index': int(next_pivot['Index']) if next_pivot else None,
                'Date': next_pivot['Date'] if next_pivot else None,
                'Price': float(next_pivot['Price']) if next_pivot else None,
                'Slope': float(next_slope) if next_slope else None,
                'PriceDiff': float(next_price_diff) if next_price_diff else None,
                'RelativeDiff': float(next_relative_diff)
            }
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

    def get_price_state(self, current_candle, previous_candle, trend_line, threshold, price_on_line):
        current_low, current_high, current_close = current_candle['Low'], current_candle['High'], current_candle[
            'Close']
        previous_low, previous_high = previous_candle['Low'], previous_candle['High']

        is_above = current_close > price_on_line
        is_below = current_close < price_on_line

        if abs(current_close - price_on_line) <= threshold:
            return 'At_Level'

        if is_below:  # 가격이 추세선 아래에 있을 때 (잠재적 저항)
            if previous_high >= price_on_line - threshold:  # 이전 캔들이 레벨에 닿았었고
                if current_high < price_on_line and current_candle['Close'] < current_candle[
                    'Open']:  # 현재 캔들이 하락하며 레벨 아래에 있다면
                    return 'Pullback'
                elif current_high > price_on_line + threshold:  # 현재 캔들이 레벨을 상향 돌파했다면
                    return 'Breakout_Up'

        elif is_above:  # 가격이 추세선 위에 있을 때 (잠재적 지지)
            if previous_low <= price_on_line + threshold:  # 이전 캔들이 레벨에 닿았었고
                if current_low > price_on_line and current_candle['Close'] > current_candle[
                    'Open']:  # 현재 캔들이 상승하며 레벨 위에 있다면
                    return 'Bounce'
                elif current_low < price_on_line - threshold:  # 현재 캔들이 레벨을 하향 돌파했다면
                    return 'Breakout_Down'

        # 그 외의 경우
        return 'Above' if is_above else 'Below'

    def check_current_price_against_trend_lines(self, df, trend_lines, symbol, interval):
        if len(df) < 2:
            return trend_lines

        current_candle = df.iloc[-1]
        previous_candle = df.iloc[-2]
        current_time = current_candle['Open Time']  # 현재 캔들의 시작 시간을 사용

        for trend_line in trend_lines:
            trend_line_id = trend_line['id']
            price_on_line = trend_line['CurrentPrice']
            threshold = 0.001 * price_on_line

            # 이전 상태와 마지막 업데이트 시간 가져오기
            previous_data = self.previous_states.get(trend_line_id, ('', None))
            if isinstance(previous_data, tuple) and len(previous_data) == 2:
                previous_state, last_update = previous_data
            else:
                previous_state, last_update = '', None

            # candle_start_time = self.get_candle_start_time(current_time, interval)

            # 새로운 캔들이 시작되었는지 확인
            if last_update is None or current_time > last_update:
                current_state = self.get_price_state(current_candle, previous_candle, trend_line, threshold,
                                                     price_on_line)

                trend_strength = 'Neutral'
                if previous_state != current_state:
                    if current_state == 'Bounce':
                        trend_strength = 'Bullish'
                    elif current_state == 'Pullback':
                        trend_strength = 'Bearish'
                    elif current_state == 'Breakout_Up':
                        trend_strength = 'Very Bullish'
                    elif current_state == 'Breakout_Down':
                        trend_strength = 'Very Bearish'
                    elif current_state == 'At_Level':
                        trend_strength = 'Consolidating'

                    # 상태 업데이트 및 마지막 업데이트 시간 저장
                self.previous_states[trend_line_id] = (current_state, current_time)

                trend_line.update({
                    'CurrentPrice': price_on_line,
                    'Difference': current_candle['Close'] - price_on_line,
                    'PreviousState': previous_state,
                    'CurrentState': current_state,
                    'TrendStrength': trend_strength,
                    'threshold': threshold,
                    'CurrentRole': 'Resistance' if price_on_line > current_candle['Close'] else 'Support',
                    'symbol': symbol,
                    'interval': interval,
                    'LastUpdate': current_time.isoformat()
                })

        return trend_lines

    # def check_current_price_against_trend_lines(self, df, trend_lines, symbol, interval):
    #     if len(df) < 2:
    #         return trend_lines  # 최소 2개의 캔들이 필요합니다
    #
    #     current_candle = df.iloc[-1]
    #     previous_candle = df.iloc[-2]
    #
    #     for trend_line in trend_lines:
    #         trend_line_id = trend_line['id']
    #
    #         price_on_line = trend_line['CurrentPrice']
    #         threshold = 0.001 * price_on_line  # 0.1% 임계값
    #         current_state = self.get_price_state(current_candle, previous_candle, trend_line, threshold, price_on_line)
    #
    #         previous_state = self.previous_states.get(trend_line_id, '')  # 기본값으로 빈 문자열 사용
    #         print(f"previous_state {previous_state}")
    #         print("-------------------")
    #
    #         trend_strength = 'Neutral'
    #         if previous_state != current_state:
    #             if current_state == 'Bounce':
    #                 trend_strength = 'Bullish'
    #             elif current_state == 'Pullback':
    #                 trend_strength = 'Bearish'
    #             elif current_state == 'Breakout_Up':
    #                 trend_strength = 'Very Bullish'
    #             elif current_state == 'Breakout_Down':
    #                 trend_strength = 'Very Bearish'
    #             elif current_state == 'At_Level':
    #                 trend_strength = 'Consolidating'
    #
    #         self.previous_states[trend_line_id] = current_state
    #
    #         # 기존 trend_line 딕셔너리에 새로운 정보 추가
    #         # 기존 trend_line 딕셔너리에 새로운 정보 추가
    #         trend_line.update({
    #             'CurrentPrice': price_on_line,
    #             'Difference': current_candle['Close'] - price_on_line,
    #             'PreviousState': previous_state,
    #             'CurrentState': current_state,
    #             'TrendStrength': trend_strength,
    #             'threshold': threshold,
    #             'CurrentRole': 'Resistance' if price_on_line > current_candle['Close'] else 'Support',
    #             'symbol': symbol,
    #             'interval': interval,
    #             'LastUpdate': candle_start_time.isoformat()
    #
    #         })
    #
    #     return trend_lines

    def get_candle_start_time(self, current_time, interval):
        from datetime import datetime, timezone, timedelta

        timestamp = int(current_time.timestamp())
        if interval == '1m':
            return datetime.fromtimestamp(timestamp - (timestamp % 60), timezone.utc)
        elif interval == '5m':
            return datetime.fromtimestamp(timestamp - (timestamp % 300), timezone.utc)
        elif interval == '15m':
            return datetime.fromtimestamp(timestamp - (timestamp % 900), timezone.utc)
        elif interval == '30m':
            return datetime.fromtimestamp(timestamp - (timestamp % 1800), timezone.utc)
        elif interval == '1h':
            return datetime.fromtimestamp(timestamp - (timestamp % 3600), timezone.utc)
        elif interval == '2h':
            return datetime.fromtimestamp(timestamp - (timestamp % 7200), timezone.utc)
        elif interval == '4h':
            return datetime.fromtimestamp(timestamp - (timestamp % 14400), timezone.utc)
        elif interval == '6h':
            return datetime.fromtimestamp(timestamp - (timestamp % 21600), timezone.utc)
        elif interval == '1d':
            return datetime.fromtimestamp(timestamp - (timestamp % 86400), timezone.utc)
        elif interval == '3d':
            return datetime.fromtimestamp(timestamp - (timestamp % 259200), timezone.utc)
        elif interval == '1w':
            return datetime.fromtimestamp(timestamp - (timestamp % 604800), timezone.utc)
        elif interval == '1M':
            # 월의 경우 정확한 계산이 복잡하므로 대략적으로 30일로 계산
            return datetime.fromtimestamp(timestamp - (timestamp % 2592000), timezone.utc)
        else:
            return current_time

    @staticmethod
    def index_pivot_points(pivot_points: List[Dict]) -> Dict[str, List[Dict]]:
        indexed = defaultdict(list)
        for pivot in pivot_points:
            indexed[pivot['Type']].append(pivot)
        for type_pivots in indexed.values():
            type_pivots.sort(key=lambda p: p['Price'], reverse=True)
        return indexed

    def select_top_trend_lines(self, trend_lines, pivot_points, historical_extremes):
        # def find_steepest_pivot_for_trend_line(trend_line: Dict):
        #     prev_slope = abs(trend_line['PrevPivot']['Slope']) if trend_line['PrevPivot'] and trend_line['PrevPivot'][
        #         'Slope'] is not None else 0
        #     next_slope = abs(trend_line['NextPivot']['Slope']) if trend_line['NextPivot'] and trend_line['NextPivot'][
        #         'Slope'] is not None else 0
        #
        #     if prev_slope > next_slope:
        #         return trend_line['PrevPivot'], prev_slope
        #     else:
        #         return trend_line['NextPivot'], next_slope
        def find_largest_price_diff_pivot(trend_line: Dict):
            prev_diff = trend_line['PrevPivot']['RelativeDiff'] if trend_line['PrevPivot'] else 0
            next_diff = trend_line['NextPivot']['RelativeDiff'] if trend_line['NextPivot'] else 0

            if prev_diff > next_diff:
                return 'PrevPivot', trend_line['PrevPivot'], prev_diff
            else:
                return 'NextPivot', trend_line['NextPivot'], next_diff

        def process_trend_lines(lines,
                                reverse_order: bool = False,
                                top_n: int = 5,
                                reference_time: Optional[pd.Timestamp] = None,
                                is_long_term: bool = False):
            if reference_time is None:
                reference_time = pd.Timestamp.now(tz='UTC')
            elif reference_time.tzinfo is None:
                reference_time = reference_time.tz_localize('UTC')

            for line in lines:
                pivot_type, largest_diff_pivot, max_diff = find_largest_price_diff_pivot(line)
                line['LargestDiffPivotType'] = pivot_type
                line['LargestDiffPivot'] = largest_diff_pivot
                line['MaxRelativePriceDiff'] = max_diff

                start_time = pd.Timestamp(line['StartDate'])
                if start_time.tzinfo is None:
                    start_time = start_time.tz_localize('UTC')

                time_diff = (reference_time - start_time).total_seconds()
                line['CurrentPrice'] = line['Slope'] * time_diff + line['StartPrice']
                line['CurrentTime'] = reference_time

                # print(
                #     f"Debug: StartTime={start_time}, TimeDiff={time_diff}, Slope={line['Slope']}, StartPrice={line['StartPrice']}, CurrentPrice={line['CurrentPrice']}")

            sorted_lines = sorted(lines, key=lambda x: x['MaxRelativePriceDiff'], reverse=True)[:top_n]

            for rank, line in enumerate(sorted_lines, start=1):
                line['Importance'] = 5 if is_long_term else rank

            # 중요도(순위)를 기준으로 정렬
            result = sorted(sorted_lines, key=lambda x: x['Importance'])[:top_n]

            return result

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
        # 각 분류별로 상위 선택
        current_time = pd.Timestamp.now(tz='UTC')

        top_recent_steep_high = process_trend_lines(recent_steep_high, reverse_order=True, top_n=5,
                                                    reference_time=current_time)
        top_recent_steep_low = process_trend_lines(recent_steep_low, reverse_order=True, top_n=5,
                                                   reference_time=current_time)
        top_long_term_high = process_trend_lines(long_term_high, reverse_order=True, top_n=1,
                                                 reference_time=current_time, is_long_term=True)
        top_long_term_low = process_trend_lines(long_term_low, reverse_order=True, top_n=1, reference_time=current_time,
                                                is_long_term=True)
        # 빈 배열 처리 (필요에 따라 기본값을 None 또는 다른 값으로 변경)
        top_recent_steep_high = top_recent_steep_high if top_recent_steep_high else []
        top_recent_steep_low = top_recent_steep_low if top_recent_steep_low else []
        top_long_term_high = top_long_term_high if top_long_term_high else []
        top_long_term_low = top_long_term_low if top_long_term_low else []

        return {
            'RecentSteepHigh': top_recent_steep_high,
            'RecentSteepLow': top_recent_steep_low,
            'LongTermHigh': top_long_term_high,
            'LongTermLow': top_long_term_low
        }


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


# scalping/views.py


# GET http://127.0.0.1:8000/api/v1/binanceData/llm-bitcoin-data/?all_last=true
# GET http://127.0.0.1:8000/api/v1/binanceData/llm-bitcoin-data/?all_last=false&symbol=BTCUSDT

class BinanceLLMChartDataAPIView(APIView):
    # 미리 정의된 다양한 심볼 리스트
    # PREDEFINED_SYMBOLS = [
    #     'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT',
    #     'DOTUSDT', 'BCHUSDT', 'LTCUSDT', 'EOSUSDT', 'SOLUSDT',
    #     'TRXUSDT', 'AAVEUSDT', 'HBARUSDT', 'NEOUSDT', 'ONTUSDT',
    #     'ATOMUSDT', 'VETUSDT', 'THETAUSDT', 'ALGOUSDT', 'UNIUSDT'
    # ]
    PREDEFINED_SYMBOLS = [
        'BTCUSDT', 'ETHUSDT','AAVEUSDT', 'CELOUSDT', 'LINKUSDT', 'MKRUSDT',
        'SOLUSDT', 'STXUSDT', 'UMAUSDT', 'UNIUSDT', 'XLMUSDT', 'XRPUSDT',
        'BNBUSDT', 'LDOUSDT', 'OPUSDT', 'SUIUSDT', 'WLDUSDT'
    ]
    def calculate_indicators(self, data: Dict[str, List[float]]) -> pd.DataFrame:
        if not data or not data.get('close'):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        # UltimateRSIAnalyzer와 SqueezeMomentumIndicator는 커스텀 클래스으로 가정
        analyzer = UltimateRSIAnalyzer(df, length=9, smoType1='RMA', smoType2='EMA', smooth=7)
        df = analyzer.get_dataframe()
        indicator = SqueezeMomentumIndicator(df)
        df = indicator.get_dataframe()
        return df

    def get_extended_kline_data(self, symbol: str, interval: str, total_candles: int = 2000) -> List[List[Any]]:
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

    def get_bitcoin_data(self, symbol: str) -> Dict[str, Any]:
        try:
            fifteen_minutes_candles = self.get_extended_kline_data(symbol, '15m')
            one_hour_candles = self.get_extended_kline_data(symbol, '1h')

            two_hour_candles = self.get_extended_kline_data(symbol, '2h')
            one_day_candles = self.get_extended_kline_data(symbol, '1d')
            one_week_candles = self.get_extended_kline_data(symbol, '1w')

            def process_candles(candles: List[List[Any]], timeframe: str) -> List[Dict[str, Any]]:
                df = pd.DataFrame(candles, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                    'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume',
                    'taker_buy_quote_asset_volume', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

                # 인디케이터 계산
                df = self.calculate_indicators(df.to_dict(orient='list'))
                df = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)
                df['timestamp'] = df['timestamp'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

                df = pd.DataFrame(df).tail(10)

                records = df.to_dict(orient='records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value) or value in [np.inf, -np.inf]:
                            record[key] = None

                return records

            # data = {
            #     '1h': process_candles(one_hour_candles, '1h'),
            #     '2h': process_candles(two_hour_candles, '2h'),
            #     '1d': process_candles(one_day_candles, '1d'),
            #     '1w': process_candles(one_week_candles, '1w')
            # }
            data = {
                '15m': process_candles(fifteen_minutes_candles, '15m'),
                '1h': process_candles(one_hour_candles, '1h'),
                '2h': process_candles(two_hour_candles, '2h'),
                '1d': process_candles(one_day_candles, '1d'),
            }
            return data

        except Exception as e:
            print(f"An error occurred: {e}")
            return {}

    def get_all_last_data(self, request) -> Response:
        try:
            long_symbols = []
            short_symbols = []

            # 멀티스레딩을 사용하여 심볼 병렬 처리
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(self.get_trading_symbols, symbol): symbol for symbol in
                           self.PREDEFINED_SYMBOLS}
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]
                    try:
                        trading_records = future.result()
                        if not trading_records:
                            continue

                        # 모든 interval이 롱 조건을 만족하는지 검사
                        if all(
                                record['rsi'] > record['rsi_signal'] and
                                record['squeeze_color'] and record['squeeze_color'].lower() in {'lime', 'maroon'}
                                for record in trading_records.values()
                        ):
                            long_symbols.append(symbol)
                        # 모든 interval이 숏 조건을 만족하는지 검사
                        elif all(
                                record['rsi'] < record['rsi_signal'] and
                                record['squeeze_color'] and record['squeeze_color'].lower() in {'red', 'green'}
                                for record in trading_records.values()
                        ):
                            short_symbols.append(symbol)

                    except Exception as e:
                        print(f"Error processing symbol {symbol}: {e}")

            # 중복 제거
            long_symbols = list(set(long_symbols))
            short_symbols = list(set(short_symbols))

            # Trading Summary 저장
            BinanceTradingSummary.objects.create(
                long_symbols=long_symbols,
                short_symbols=short_symbols
            )

            return Response({
                "long_symbols": long_symbols,
                "short_symbols": short_symbols
            })

        except Exception as e:
            print(f"Error processing all_last=true request: {e}")
            return Response(
                {'error': f'An unexpected error occurred: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def get_trading_symbols(self, symbol: str) -> Dict[str, Any]:
        try:
            # 원하는 인터벌로 데이터 가져오기
            two_hour_candles = self.get_extended_kline_data(symbol, '2h')
            one_day_candles = self.get_extended_kline_data(symbol, '1d')
            one_week_candles = self.get_extended_kline_data(symbol, '1w')

            def process_candles(candles: List[List[Any]], timeframe: str) -> Dict[str, Any]:
                df = pd.DataFrame(candles, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

                # 인디케이터 계산
                df_with_indicators = self.calculate_indicators(df.to_dict(orient='list'))

                if 'timestamp' not in df_with_indicators.columns:
                    df_with_indicators.reset_index(inplace=True)
                    df_with_indicators.rename(columns={'index': 'timestamp'}, inplace=True)

                # 마지막 한 행만 추출
                last_row = df_with_indicators.iloc[-1].to_dict()

                # NaN 처리: NaN을 None으로 대체
                last_row_clean = {k: (v if pd.notna(v) else None) for k, v in last_row.items()}

                # 조건 체크: RSI > RSI_signal 및 SqueezeColor가 'lime' 또는 'maroon'
                if last_row_clean.get('RSI', 0) > last_row_clean.get('RSI_signal', 0):
                    if last_row_clean.get('SqueezeColor', '').lower() in {'lime', 'maroon'}:
                        position = 'long'
                elif last_row_clean.get('RSI', 0) < last_row_clean.get('RSI_signal', 0):
                    if last_row_clean.get('SqueezeColor', '').lower() in {'red', 'green'}:
                        position = 'short'
                else:
                    position = 'hold'

                return {
                    'symbol': symbol,
                    'interval': timeframe,
                    'rsi': last_row_clean.get('RSI'),
                    'rsi_signal': last_row_clean.get('RSI_signal'),
                    'squeeze_color': last_row_clean.get('SqueezeColor'),
                    'position': position
                }

            trading_records = {}
            for interval, candles in [('2h', two_hour_candles), ('1d', one_day_candles), ('1w', one_week_candles)]:
                if not candles:
                    continue
                record = process_candles(candles, interval)
                trading_records[interval] = record

            return trading_records

        except Exception as e:
            print(f"An error occurred while processing symbol {symbol}: {e}")
            return {}

    def get(self, request):
        all_last = request.GET.get('all_last', 'false').lower() == 'true'
        if all_last:
            return self.get_all_last_data(request)
        else:
            # 기존 로직 유지
            symbol = request.GET.get('symbol', 'BTCUSDT')
            data = self.get_bitcoin_data(symbol)
            if not data:
                return Response({"error": "Failed to fetch data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(data)



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
#     def get_extended_kline_data(self, symbol, interval, total_candles=2000):
#         binance_api_url = "https://api.binance.com/api/v3/klines"
#         limit = 500  # Binance API limit per request
#         all_candles = []
#
#         while len(all_candles) < total_candles:
#             params = {
#                 'symbol': symbol,
#                 'interval': interval,
#                 'limit': limit
#             }
#             if all_candles:
#                 params['endTime'] = all_candles[0][0] - 1
#
#             try:
#                 response = requests.get(binance_api_url, params=params, timeout=10)
#                 response.raise_for_status()
#                 candles = response.json()
#                 if not candles:
#                     break
#                 all_candles = candles + all_candles
#             except requests.exceptions.RequestException as e:
#                 print(f"Error fetching data for {symbol}, {interval}: {e}")
#                 break
#
#         return all_candles[:total_candles]

    # def get_bitcoin_data(self, symbol):
    #     try:
    #         fifteen_min_candles = self.get_extended_kline_data(symbol, '1h')
    #         thirty_min_candles = self.get_extended_kline_data(symbol, '2h')
    #         hourly_candles = self.get_extended_kline_data(symbol, '1d')
    #         daily_candles = self.get_extended_kline_data(symbol, '1w')
    #
    #         # fifteen_min_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE, limit=500)
    #         # thirty_min_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE, limit=500)
    #         # hourly_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, limit=500)
    #         # daily_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, limit=500)
    #
    #         # end_date = datetime.now()
    #         # start_date_15min = end_date - timedelta(days=7)  # Last 7 days
    #         # start_date_30min = end_date - timedelta(days=15)  # Last 14 days
    #         # start_date_hourly = end_date - timedelta(days=30)  # Last 30 days
    #         # start_date_daily = end_date - timedelta(days=730)  # Last 365 days
    #         #
    #         # print("Fetching data...")
    #         #
    #         # fifteen_min_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE,
    #         #                                                         start_date_15min.strftime("%d %b %Y %H:%M:%S"),
    #         #                                                         end_date.strftime("%d %b %Y %H:%M:%S"))
    #         #
    #         # thirty_min_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE,
    #         #                                                        start_date_30min.strftime("%d %b %Y %H:%M:%S"),
    #         #                                                        end_date.strftime("%d %b %Y %H:%M:%S"))
    #         #
    #         # hourly_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR,
    #         #                                                    start_date_hourly.strftime("%d %b %Y %H:%M:%S"),
    #         #                                                    end_date.strftime("%d %b %Y %H:%M:%S"))
    #         #
    #         # daily_candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY,
    #         #                                                   start_date_daily.strftime("%d %b %Y %H:%M:%S"),
    #         #                                                   end_date.strftime("%d %b %Y %H:%M:%S"))
    #
    #         def process_candles(candles, timeframe):
    #             df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    #                                                 'quote_asset_volume', 'number_of_trades',
    #                                                 'taker_buy_base_asset_volume',
    #                                                 'taker_buy_quote_asset_volume', 'ignore'])
    #             df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    #             df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    #             df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(
    #                 float)
    #
    #             # Calculate additional technical indicators
    #             # for ma in [5, 10, 20, 50, 100, 200]:
    #             #     df[f"MA{ma}"] = df["close"].rolling(window=ma).mean()
    #             #
    #             # # Bollinger Bands
    #             # period = 20
    #             # multiplier = 2.0
    #             # df["MA"] = df["close"].rolling(window=period).mean()
    #
    #             analyzer = UltimateRSIAnalyzer(df, length=14, smoType1='RMA', smoType2='EMA', smooth=14)
    #             df = analyzer.get_dataframe()
    #             indicator = SqueezeMomentumIndicator(df)
    #             df = indicator.get_dataframe()
    #
    #             # df["STD"] = df["close"].rolling(window=period).std()
    #             # df["Upper"] = df["MA"] + (df["STD"] * multiplier)
    #             # df["Lower"] = df["MA"] - (df["STD"] * multiplier)
    #
    #             # MACD
    #             # exp1 = df['close'].ewm(span=12, adjust=False).mean()
    #             # exp2 = df['close'].ewm(span=26, adjust=False).mean()
    #             # df['MACD'] = exp1 - exp2
    #             # df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    #
    #             # StochasticRSI(df)
    #             # RSIAnalyzer(df)
    #             # 시간 프레임별 일목균형표 설정값 정의
    #             # if timeframe == '15min':
    #             #     ichimoku_settings = {'tenkan': 7, 'kijun': 22, 'senkou_b': 44, 'displacement': 22}
    #             # elif timeframe == '30min':
    #             #     ichimoku_settings = {'tenkan': 9, 'kijun': 26, 'senkou_b': 52, 'displacement': 26}
    #             # elif timeframe == '2hourly':
    #             #     ichimoku_settings = {'tenkan': 9, 'kijun': 26, 'senkou_b': 52, 'displacement': 26}
    #             # elif timeframe == '6hourly':
    #             #     ichimoku_settings = {'tenkan': 20, 'kijun': 60, 'senkou_b': 120, 'displacement': 30}
    #             # else:
    #             #     # 기본 설정값
    #             #     ichimoku_settings = {'tenkan': 9, 'kijun': 26, 'senkou_b': 52, 'displacement': 26}
    #             #
    #             #
    #             # ichimoku = IchimokuIndicator(df, **ichimoku_settings)
    #             # ichimoku_df = ichimoku.get_ichimoku()
    #             # df = pd.merge(df, ichimoku_df, on='timestamp', how='left')
    #
    #             import numpy as np
    #             df = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)
    #             df['timestamp'] = df['timestamp'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
    #
    #             records = df.to_dict(orient='records')
    #             for record in records:
    #                 for key, value in record.items():
    #                     if pd.isna(value) or value in [np.inf, -np.inf]:
    #                         record[key] = None
    #
    #             return records
    #
    #         return {
    #             '1hour': process_candles(fifteen_min_candles, '1h'),
    #             '2hour': process_candles(thirty_min_candles, '2h'),
    #             'daily': process_candles(hourly_candles, '1d'),
    #             'weekly': process_candles(daily_candles, '1w')
    #         }
    #
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         return None

class BinanceScalpingDataView(APIView):

    def fetch_fear_and_greed_index(self, limit=30, date_format='') -> float:
        """
        Fetches the Fear and Greed Index from alternative.me API.

        Parameters:
        - limit (int): Number of results to return. Default is 1.
        - date_format (str): Date format ('us', 'cn', 'kr', 'world'). Default is '' (unixtime).

        Returns:
        - float: The Fear and Greed Index value
        """
        try:
            base_url = "https://api.alternative.me/fng/"
            params = {
                'limit': limit,
                'format': 'json',
                'date_format': date_format
            }
            response = requests.get(base_url, params=params)
            data = response.json()['data'][0]  # Get the latest data
            return float(data['value'])
        except Exception as e:
            print(f"Error fetching fear/greed index: {e}")
            return 50  # Default neutral value



    def get_market_conditions(self, df: pd.DataFrame) -> dict:
        """Calculate additional market condition indicators"""
        try:
            # Price change calculation - 가능한 최근 데이터로 계산
            available_periods = len(df)
            if available_periods < 2:
                price_change = 0.0
            else:
                # 최근 가격 변화율 계산
                latest_close = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[0]  # 사용 가능한 첫 데이터
                price_change = ((latest_close - prev_close) / prev_close) * 100

            # Volatility calculation

            middle_band, upper_band, lower_band = self.calculate_bollinger_bands(df)
            current_volatility = ((upper_band.iloc[-1] - lower_band.iloc[-1]) / df['Close'].iloc[-1]) * 100

            # Volume analysis
            avg_volume = df['Volume'].rolling(window=30).mean().iloc[-1]
            volume_ratio = (df['Volume'].iloc[-1] / avg_volume)

            return {
                'price_change_24h': float(price_change),
                'volatility': float(current_volatility),
                'volume_ratio': float(volume_ratio)
            }
        except Exception as e:
            print(f"Error calculating market conditions: {e}")
            return {
                'price_change_24h': 0.0,
                'volatility': 0.0,
                'volume_ratio': 1.0
            }

    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, num_std: int = 2) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands

        Parameters:
        - df: DataFrame with price data
        - period: Moving average period (default: 20)
        - num_std: Number of standard deviations (default: 2)

        Returns:
        - Tuple of (Middle Band, Upper Band, Lower Band)
        """
        # Calculate middle band (20-day SMA)
        middle_band = df['Close'].rolling(window=period).mean()  # 'close' -> 'Close'

        # Calculate standard deviation
        std_dev = df['Close'].rolling(window=period).std()  # 'close' -> 'Close'

        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)

        return middle_band, upper_band, lower_band
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[
        pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        high = df['High'].rolling(k_period).max()
        low = df['Low'].rolling(k_period).min()
        k = 100 * (df['Close'] - low) / (high - low)
        d = k.rolling(d_period).mean()
        return k, d

    def calculate_rsi(self, df: pd.DataFrame, length: int = 14) -> pd.Series:
        """
        Calculate RSI using pandas_ta

        Parameters:
        - df: DataFrame with price data
        - length: RSI period (default: 14)

        Returns:
        - Series: RSI values
        """
        try:
            rsi = ta.rsi(df['Close'], length=length)
            return rsi
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return pd.Series([np.nan] * len(df))  # 에러 시 NaN 값 반환

    def calculate_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> \
    Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Parameters:
        - df: DataFrame with 'Close' prices
        - fast_period: Short-term EMA period (default: 12)
        - slow_period: Long-term EMA period (default: 26)
        - signal_period: Signal line EMA period (default: 9)

        Returns:
        - Tuple of (MACD line, Signal line, Histogram)
        """
        # Calculate the EMAs
        fast_ema = df['Close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['Close'].ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        macd = fast_ema - slow_ema

        # Calculate Signal line
        signal = macd.ewm(span=signal_period, adjust=True).mean()

        # Calculate Histogram
        histogram = macd - signal

        return macd, signal, histogram
    def calculate_moving_averages(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate moving averages"""
        ma7 = df['Close'].rolling(window=7).mean()
        ma25 = df['Close'].rolling(window=25).mean()
        ma99 = df['Close'].rolling(window=99).mean()
        ma200 = df['Close'].rolling(window=200).mean()

        return ma7, ma25, ma99, ma200

    @method_decorator(cache_page(150))  # Cache for 1 minute for scalping
    def get(self, request, symbol: str, interval: str = '1m') -> Response:
        """Get candlestick data with technical indicators for scalping"""
        binance_api_url = "https://api.binance.com/api/v3/klines"
        limit = 1000  # 충분한 데이터 포인트

        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            response = requests.get(binance_api_url, params=params)
            response.raise_for_status()
            candles = response.json()

            # DataFrame 생성
            df = pd.DataFrame(candles, columns=[
                "Open Time", "Open", "High", "Low", "Close", "Volume",
                "Close Time", "Quote Asset Volume", "Number of Trades",
                "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
            ])

            # 데이터 타입 변환
            df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
            df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col])

            # 기술적 지표 계산
            rsi = self.calculate_rsi(df)
            macd, signal, histogram = self.calculate_macd(df)
            ma7, ma25, ma99, ma200 = self.calculate_moving_averages(df)
            middle_band, upper_band, lower_band = self.calculate_bollinger_bands(df)
            stoch_k, stoch_d = self.calculate_stochastic(df)  # 스토캐스틱 계산 추가
            fear_greed_index = self.fetch_fear_and_greed_index()  # 새로운 메서드 사용
            market_conditions = self.get_market_conditions(df)
            # 최근 30개 캔들만 사용
            recent_data = []
            for i in range(-30, 0):
                candle_data = {
                    'timestamp': df["Open Time"].iloc[i].isoformat(),
                    'open': float(df["Open"].iloc[i]),
                    'high': float(df["High"].iloc[i]),
                    'low': float(df["Low"].iloc[i]),
                    'close': float(df["Close"].iloc[i]),
                    'volume': float(df["Volume"].iloc[i]),
                    'indicators': {
                        'rsi': float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else None,
                        'macd': {
                            'macd': float(macd.iloc[i]) if not pd.isna(macd.iloc[i]) else None,
                            'signal': float(signal.iloc[i]) if not pd.isna(signal.iloc[i]) else None,
                            'histogram': float(histogram.iloc[i]) if not pd.isna(histogram.iloc[i]) else None
                        },
                        'moving_averages': {
                            'ma7': float(ma7.iloc[i]) if not pd.isna(ma7.iloc[i]) else None,
                            'ma25': float(ma25.iloc[i]) if not pd.isna(ma25.iloc[i]) else None,
                            'ma99': float(ma99.iloc[i]) if not pd.isna(ma99.iloc[i]) else None
                        },
                        'stochastic': {  # 스토캐스틱 추가
                            'k': float(stoch_k.iloc[i]) if not pd.isna(stoch_k.iloc[i]) else None,
                            'd': float(stoch_d.iloc[i]) if not pd.isna(stoch_d.iloc[i]) else None
                        },
                        'bollinger_bands': {  # 볼린저 밴드 추가
                            'middle': float(middle_band.iloc[i]) if not pd.isna(middle_band.iloc[i]) else None,
                            'upper': float(upper_band.iloc[i]) if not pd.isna(upper_band.iloc[i]) else None,
                            'lower': float(lower_band.iloc[i]) if not pd.isna(lower_band.iloc[i]) else None
                        }
                    }
                }
                recent_data.append(candle_data)

            return Response({
                'symbol': symbol,
                'interval': interval,
                'last_update': pd.Timestamp.now().isoformat(),
                'data': recent_data
            })

        except requests.RequestException as e:
            return Response(
                {'error': f'Failed to fetch data from Binance: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            return Response(
                {'error': f'An unexpected error occurred: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

# http://127.0.0.1:8000/api/v1/binanceData/upbit/?symbol=KRW-BTC
# http://127.0.0.1:8000/api/v1/binanceData/upbit/?all_last=true
class UpbitDataView(APIView):
    # 미리 정의된 20개 심볼 리스트 (예시)
    # PREDEFINED_SYMBOLS = [
    #     'KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-ADA', 'KRW-DOGE',
    #     'KRW-AAVE', 'KRW-SUI', 'KRW-PEPE', 'KRW-CTC', 'KRW-SOL',
    #     'KRW-TRX', 'KRW-HBAR', 'KRW-LINK', 'KRW-SEI', 'KRW-DOT',
    #     'KRW-IQ', 'KRW-ADA', 'KRW-VET', 'KRW-SAND', 'KRW-ALGO'
    # ]
    PREDEFINED_SYMBOLS = [
        'KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-ADA', 'KRW-DOGE',
        'KRW-DOT', 'KRW-BCH', 'KRW-LTC', 'KRW-EOS', 'KRW-SOL',
        'KRW-TRX', 'KRW-AAVE', 'KRW-HBAR', 'KRW-NEO', 'KRW-ONT',
        'KRW-ATOM', 'KRW-ADA', 'KRW-VET', 'KRW-THETA', 'KRW-ALGO'
    ]
    def calculate_indicators(self, data: Dict[str, List[float]]) -> pd.DataFrame:
        if not data or not data.get('close'):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        analyzer = UltimateRSIAnalyzer(df, length=9, smoType1='RMA', smoType2='EMA', smooth=7)
        df = analyzer.get_dataframe()
        indicator = SqueezeMomentumIndicator(df)
        df = indicator.get_dataframe()
        return df

    # def get_all_last_data(self, request):
    #     """
    #     모든 미리 정의된 KRW 마켓 심볼에 대해 마지막 한 행의 데이터를 가져오는 메서드
    #     URL 예: /api/v1/binanceData/upbit/?all_last=true
    #     """
    #     limit = 500
    #     # intervals = {
    #     #     '1hour': 'minute60',
    #     #     '4hour': 'minute240',
    #     #     '1day': 'day',
    #     #     '1week': 'week'
    #     # }
    #     intervals = {
    #         '15m': 'minute15',
    #         '1hour': 'minute60',
    #         '1day': 'day',
    #     }
    #     # 미리 정의된 심볼 리스트 사용
    #     all_symbols = self.PREDEFINED_SYMBOLS
    #     result_all_symbols = {}
    #
    #     try:
    #         for sym in all_symbols:
    #             symbol_results = {}
    #             for label, iv in intervals.items():
    #                 df = pyupbit.get_ohlcv(sym, interval=iv, count=limit)
    #                 if df is None or df.empty:
    #                     symbol_results[label] = {'error': f'No data available for {iv}'}
    #                     continue
    #
    #                 df.reset_index(inplace=True)
    #                 df.rename(columns={'index': 'timestamp'}, inplace=True)
    #                 data_dict = {
    #                     'timestamp': df['timestamp'].astype(str).tolist(),
    #                     'open': df['open'].tolist(),
    #                     'high': df['high'].tolist(),
    #                     'low': df['low'].tolist(),
    #                     'close': df['close'].tolist(),
    #                     'volume': df['volume'].tolist()
    #                 }
    #
    #                 df_with_indicators = self.calculate_indicators(data_dict)
    #
    #                 if 'timestamp' not in df_with_indicators.columns:
    #                     df_with_indicators.reset_index(inplace=True)
    #                     df_with_indicators.rename(columns={'index': 'timestamp'}, inplace=True)
    #
    #                 # 마지막 한 행만 추출
    #                 last_row = df_with_indicators.iloc[-1].to_dict()
    #                 symbol_results[label] = last_row
    #             result_all_symbols[sym] = symbol_results
    #
    #         return Response(result_all_symbols)
    #
    #     except Exception as e:
    #         return Response(
    #             {'error': f'An unexpected error occurred: {str(e)}'},
    #             status=status.HTTP_500_INTERNAL_SERVER_ERROR
    #         )



    def get_all_last_data(self, request):
        """
        모든 미리 정의된 KRW 마켓 심볼에 대해 조건을 만족하는 심볼을 필터링하고, TradingRecord 모델에 저장하는 메서드
        URL 예: /api/v1/binanceData/upbit/?all_last=true
        """
        limit = 500
        intervals = {
            '4hour': 'minute240',
            '1day': 'day',
            '1week': 'week'
        }
        all_symbols = self.PREDEFINED_SYMBOLS
        filtered_symbols = []

        try:
            for sym in all_symbols:
                all_intervals_pass = True  # 모든 인터벌이 조건을 만족하는지 여부

                for label, iv in intervals.items():
                    try:
                        df = pyupbit.get_ohlcv(sym, interval=iv, count=limit)
                        if df is None or df.empty:
                            print(f'No data available for {sym} in interval {iv}')
                            all_intervals_pass = False
                            break  # 해당 심볼은 조건을 만족할 수 없음

                        df.reset_index(inplace=True)
                        df.rename(columns={'index': 'timestamp'}, inplace=True)
                        data_dict = {
                            'timestamp': df['timestamp'].astype(str).tolist(),
                            'open': df['open'].tolist(),
                            'high': df['high'].tolist(),
                            'low': df['low'].tolist(),
                            'close': df['close'].tolist(),
                            'volume': df['volume'].tolist()
                        }

                        df_with_indicators = self.calculate_indicators(data_dict)

                        if 'timestamp' not in df_with_indicators.columns:
                            df_with_indicators.reset_index(inplace=True)
                            df_with_indicators.rename(columns={'index': 'timestamp'}, inplace=True)

                        # 마지막 한 행만 추출
                        last_row = df_with_indicators.iloc[-1].to_dict()

                        # NaN 처리: NaN을 None으로 대체
                        last_row_clean = {k: (v if pd.notna(v) else None) for k, v in last_row.items()}

                        # # 조건 체크: RSI > RSI_signal 및 SqueezeColor가 lime 또는 maroon
                        # if not (last_row_clean.get('RSI', 0) > last_row_clean.get('RSI_signal', 0)):
                        #     all_intervals_pass = False
                        #     break  # 해당 심볼은 조건을 만족할 수 없음


                        # 조건 체크: RSI > RSI_signal 및 SqueezeColor가 lime 또는 maroon
                        if not (last_row_clean.get('RSI', 0) > last_row_clean.get('RSI_signal', 0) and
                                last_row_clean.get('SqueezeColor', '').lower() in {'lime', 'maroon'}):
                            all_intervals_pass = False
                            break  # 해당 심볼은 조건을 만족할 수 없음

                    except Exception as interval_e:
                        # 로그 남기기
                        print(f'Error processing symbol {sym} for interval {label}: {str(interval_e)}')
                        all_intervals_pass = False
                        break

                if all_intervals_pass:
                    filtered_symbols.append(sym)
                    print(f'Symbol {sym} meets the criteria and is added to the list.')

            if filtered_symbols:
                # TradingRecord 모델에 저장
                TradingRecord.objects.create(symbols=filtered_symbols)
                print(f'{len(filtered_symbols)} symbols saved to TradingRecord.')
            else:
                TradingRecord.objects.create(symbols=[])
                print('No symbols met the criteria.')

            return Response({"symbols_saved": filtered_symbols})

        except Exception as e:
            # 로그 남기기
            print(f'Error processing symbols: {str(e)}')
            return Response(
                {'error': f'An unexpected error occurred: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )




    def get(self, request):
        all_last = request.GET.get('all_last', 'false').lower() == 'true'
        if all_last:
            return self.get_all_last_data(request)
        else:
            # 기존 로직
            limit = 1000
            symbol = request.GET.get('symbol', 'KRW-BTC')

            if not symbol.startswith('KRW-'):
                symbol = f'KRW-{symbol}'

            intervals = {
                '1hour': 'minute60',
                '4hour': 'minute240',
                '1day': 'day',
                '1week': 'week'
            }

            result_all_intervals = {}

            try:
                for label, iv in intervals.items():
                    df = pyupbit.get_ohlcv(symbol, interval=iv, count=limit)
                    if df is None or df.empty:
                        result_all_intervals[label] = {'error': f'No data available for {iv}'}
                        continue

                    df.reset_index(inplace=True)
                    df.rename(columns={'index': 'timestamp'}, inplace=True)

                    data_dict = {
                        'timestamp': df['timestamp'].astype(str).tolist(),
                        'open': df['open'].tolist(),
                        'high': df['high'].tolist(),
                        'low': df['low'].tolist(),
                        'close': df['close'].tolist(),
                        'volume': df['volume'].tolist()
                    }

                    df_with_indicators = self.calculate_indicators(data_dict)

                    if 'timestamp' not in df_with_indicators.columns:
                        df_with_indicators.reset_index(inplace=True)
                        df_with_indicators.rename(columns={'index': 'timestamp'}, inplace=True)

                    df_with_indicators = df_with_indicators.iloc[-30:]
                    records = df_with_indicators.to_dict(orient='records')

                    result_all_intervals[label] = records

                return Response(result_all_intervals)

            except Exception as e:
                return Response(
                    {'error': f'An unexpected error occurred: {str(e)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )


# class UpbitDataView(APIView):
#     def calculate_indicators(self, data: Dict[str, List[float]]) -> pd.DataFrame:
#         """Calculate various indicators using dictionary data and return a DataFrame."""
#         if not data or not data.get('close'):
#             # 빈 DataFrame 반환하거나 예외 처리
#             return pd.DataFrame()
#
#         # Convert to DataFrame temporarily for calculations
#         df = pd.DataFrame(data)
#
#         # 인디케이터 계산 (예: UltimateRSIAnalyzer, SqueezeMomentumIndicator)
#         analyzer = UltimateRSIAnalyzer(df, length=14, smoType1='RMA', smoType2='EMA', smooth=14)
#         df = analyzer.get_dataframe()
#         indicator = SqueezeMomentumIndicator(df)
#         df = indicator.get_dataframe()
#
#         return df
#
#     def get(self, request):
#         """Get candlestick data with technical indicators for multiple intervals."""
#         limit = 1000
#         symbol = request.GET.get('symbol', 'KRW-BTC')
#
#         try:
#             if not symbol.startswith('KRW-'):
#                 symbol = f'KRW-{symbol}'
#
#             # 원하는 interval을 딕셔너리로 정의
#             intervals = {
#                 '1hour': 'minute60',
#                 '4hour': 'minute240',
#                 '1day': 'day',
#                 '1week': 'week'
#             }
#
#             result_all_intervals = {}
#
#             for label, iv in intervals.items():
#                 df = pyupbit.get_ohlcv(symbol, interval=iv, count=limit)
#
#                 if df is None or df.empty:
#                     result_all_intervals[label] = {'error': f'No data available for {iv}'}
#                     continue
#
#                 # dict로 변환하기 전에 index를 컬럼으로 변환 후 timestamp 컬럼으로 이름 변경
#                 df.reset_index(inplace=True)
#                 df.rename(columns={'index': 'timestamp'}, inplace=True)
#
#                 # Dict 형태로 변환하기 위해 dictionary 생성
#                 data_dict = {
#                     'timestamp': df['timestamp'].astype(str).tolist(),
#                     'open': df['open'].tolist(),
#                     'high': df['high'].tolist(),
#                     'low': df['low'].tolist(),
#                     'close': df['close'].tolist(),
#                     'volume': df['volume'].tolist()
#                 }
#
#                 # 인디케이터 계산
#                 df_with_indicators = self.calculate_indicators(data_dict)
#
#                 # 다시 timestamp 컬럼이 사라졌다면 복구 (calculate_indicators에서는 index 사용)
#                 if 'timestamp' not in df_with_indicators.columns:
#                     # df_with_indicators는 calculate_indicators에서 index 사용
#                     # 원래 timestamp를 담기 위해 다시 reset_index 후 rename 필요
#                     df_with_indicators.reset_index(inplace=True)
#                     df_with_indicators.rename(columns={'index': 'timestamp'}, inplace=True)
#
#                 # 마지막 30개만 추출 (원한다면)
#                 df_with_indicators = df_with_indicators.iloc[-30:]
#
#                 # 각 행을 하나의 딕셔너리로 변환
#                 records = df_with_indicators.to_dict(orient='records')
#
#                 # 각 레코드는 다음 형태를 갖게 됨:
#                 # [{'timestamp': '2023-08-20 00:00:00', 'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ..., 'UltimateRSI':..., 'SqueezeMomentum':...}, ...]
#
#                 result_all_intervals[label] = records
#
#             return Response(result_all_intervals)
#
#         except Exception as e:
#             return Response(
#                 {'error': f'An unexpected error occurred: {str(e)}'},
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )


# class UpbitDataView(APIView):
#     def calculate_indicators(self, data: Dict[str, List[float]]) -> Dict[str, List[Any]]:
#         """Calculate various indicators using dictionary data."""
#         if not data or not data.get('close'):
#             return Response({'error': 'No data available'}, status=status.HTTP_404_NOT_FOUND)
#
#         # Convert to DataFrame temporarily for calculations
#         df = pd.DataFrame(data)
#         # result = {
#         #     'timestamp': data['index'],
#         #     'open': data['open'],
#         #     'high': data['high'],
#         #     'low': data['low'],
#         #     'close': data['close'],
#         #     'volume': data['volume'],
#         # }
#         #
#         # # Calculate indicators
#         # result['SMA_10'] = self._convert_series_to_list(ta.sma(df['close'], length=10))
#         # result['EMA_10'] = self._convert_series_to_list(ta.ema(df['close'], length=10))
#         # result['RSI_14'] = self._convert_series_to_list(ta.rsi(df['close'], length=14))
#         #
#         # # MACD
#         # macd, signal, histogram = self.calculate_macd(df)
#         # result['MACD'] = self._convert_series_to_list(macd)
#         # result['Signal_Line'] = self._convert_series_to_list(signal)
#         # result['MACD_Histogram'] = self._convert_series_to_list(histogram)
#         #
#         # # Moving Averages
#         # result['MA7'] = self._convert_series_to_list(df['close'].rolling(window=7, min_periods=1).mean())
#         # result['MA25'] = self._convert_series_to_list(df['close'].rolling(window=25, min_periods=1).mean())
#         # result['MA99'] = self._convert_series_to_list(df['close'].rolling(window=99, min_periods=1).mean())
#         #
#         # # Bollinger Bands
#         # middle, upper, lower = self.calculate_bollinger_bands(df)
#         # result['Middle_Band'] = self._convert_series_to_list(middle)
#         # result['Upper_Band'] = self._convert_series_to_list(upper)
#         # result['Lower_Band'] = self._convert_series_to_list(lower)
#         analyzer = UltimateRSIAnalyzer(df, length=14, smoType1='RMA', smoType2='EMA', smooth=14)
#         df = analyzer.get_dataframe()
#         indicator = SqueezeMomentumIndicator(df)
#         df = indicator.get_dataframe()
#
#
#
#         return df
#
#     # def _convert_series_to_list(self, series: pd.Series) -> List[float]:
#     #     """Convert pandas Series to list, handling NaN values."""
#     #     return [None if pd.isna(x) else float(x) for x in series]
#     #
#     # def calculate_macd(self, df: pd.DataFrame, fast_period: int = 6, slow_period: int = 13, signal_period: int = 5) -> \
#     #         Tuple[pd.Series, pd.Series, pd.Series]:
#     #     fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
#     #     slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
#     #     macd = fast_ema - slow_ema
#     #     signal = macd.ewm(span=signal_period, adjust=False).mean()
#     #     histogram = macd - signal
#     #     return macd, signal, histogram
#     #
#     # def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, num_std: int = 2) -> Tuple[
#     #     pd.Series, pd.Series, pd.Series]:
#     #     middle_band = df['close'].rolling(window=period).mean()
#     #     std_dev = df['close'].rolling(window=period).std()
#     #     upper_band = middle_band + (std_dev * num_std)
#     #     lower_band = middle_band - (std_dev * num_std)
#     #     return middle_band, upper_band, lower_band
#
#     @method_decorator(cache_page(60))
#     def get(self, request, symbol: str = 'KRW-BTC', interval: str = 'minute1') -> Response:
#         """Get candlestick data with technical indicators for scalping from Upbit."""
#         limit = 1000
#
#         try:
#             if not symbol.startswith('KRW-'):
#                 symbol = f'KRW-{symbol}'
#
#             # Get data from Upbit
#             df = pyupbit.get_ohlcv(symbol, interval=interval, count=limit)
#
#             if df is None or df.empty:
#                 return Response({'error': 'No data available'}, status=status.HTTP_404_NOT_FOUND)
#
#             # Convert DataFrame to dictionary
#             data_dict = {
#                 'index': df.index.astype(str).tolist(),
#                 'open': df['open'].tolist(),
#                 'high': df['high'].tolist(),
#                 'low': df['low'].tolist(),
#                 'close': df['close'].tolist(),
#                 'volume': df['volume'].tolist()
#             }
#
#             # Calculate indicators
#             result_dict = self.calculate_indicators(data_dict)
#
#             # Select last 30 data points
#             for key in result_dict:
#                 result_dict[key] = result_dict[key][-30:]
#
#             return Response(result_dict)
#
#         except Exception as e:
#             return Response(
#                 {'error': f'An unexpected error occurred: {str(e)}'},
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )
#


# /api/v1/binanceData/stockData/?all_last=true
# https://gridtrade.one/api/v1/binanceData/stockData/?all_last=true
#http://127.0.0.1:8000/api/v1/binanceData/stockData/?symbol=AAPL
# https://www.myfinpl.com/investment/stock/sector/consumer-defensive#goog_rewarded
class stockDataView(APIView):

    # PREDEFINED_SYMBOLS = [
    #     'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    #     'META', 'NVDA', 'NFLX', 'BRK-B', 'JNJ'
    # ]
    PREDEFINED_SYMBOLS = [
        # 나스닥 상장 주요 기업 (IT, 반도체, 소비재, 헬스케어)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'PYPL',
        'CSCO', 'PEP', 'CMCSA', 'INTC', 'QCOM', 'TXN', 'AVGO', 'AMD', 'ISRG', 'MDLZ',
        'BKNG', 'VRTX', 'HON', 'TMUS', 'LRCX', 'GILD', 'ADSK', 'AMAT', 'MRNA', 'ASML', 'NET',
        'DOCU', 'OKTA', 'ZS', 'MDB', 'SHOP',

        # 뉴욕증권거래소(NYSE) 주요 기업 (금융, 에너지, 산업, 소비재)
        'BRK-B', 'JNJ', 'V', 'JPM', 'WMT', 'PG', 'MA', 'XOM', 'CVX', 'KO',
        'DIS', 'MCD', 'ABBV', 'T', 'PFE', 'BAC', 'HD', 'UNH', 'LLY', 'NKE',
        'CRM', 'IBM', 'GE', 'UPS', 'BA', 'MMM', 'CAT', 'SPGI', 'BLK', 'TMO',
        'COST', 'WBA', 'LOW', 'CVS', 'EL', 'MO', 'PM', 'KHC', 'CL', 'KMB',
        'FDX', 'RTX', 'DHR',

        # 에너지 및 산업 분야 (대형 에너지 기업 및 자동차)
        'SLB', 'HAL', 'PSX', 'OXY', 'EOG', 'F', 'GM', 'DE', 'NOC', 'LMT',

        # 클라우드 및 SaaS (뉴욕거래소와 나스닥 혼합)
        # -> 여기서 'MDB', 'CRM' 두 번째 등장분 제거
        'ORCL', 'SAP', 'INTU', 'NOW', 'ADP', 'SNOW', 'ZM', 'TWLO',
        'CRWD', 'DDOG',

        # 헬스케어 및 제약
        # -> 여기서 'TMO' 두 번째 등장분 제거
        'MRK', 'BMY', 'ABT', 'ZBH', 'SYK', 'BDX', 'CI',

        # 소비재 및 통신 (경기방어주 추가)
        # -> 여기서 'KO', 'MCD', 'PG', 'CL', 'KMB', 'EL', 'T', 'WMT', 'COST' 제거
        'PEP', 'SBUX', 'YUM', 'DPZ', 'VZ', 'TMUS', 'TGT', 'GEV',

        # 추가된 경기방어주
        # -> 여기서 'MDLZ' 제거
        'KHC', 'GIS', 'HSY', 'CLX', 'HRL', 'CPB', 'CHD', 'MKC', 'SJM',

        'CONE', 'FATE', 'EXAS', 'ONON', 'SPLK',
        'REGN', 'BIIB', 'INCY', 'ALXN',
        'ROST', 'TAP', 'CPRT', 'KORS',
        'HES', 'PXD', 'APA', 'OIL',
        'COF', 'MTB', 'CBOE', 'FITB', 'RF',
        'FLIR', 'VRSK', 'IFF', 'NVR', 'DOV',
        'TEAM', 'WDAY', 'ANET', 'PTON',

        # 나스닥 신규 추가
        'ABNB', 'HOOD', 'COIN', 'AFRM', 'SOFI', 'DUOL', 'BMBL', 'GTLB',

        # 뉴욕증권거래소 신규 추가
        # -> 여기서 'AMAT', 'DIS' 제거
        'TTAN', 'PONY', 'NOTE', 'UBER', 'SPOT',
        'RBLX', 'U', 'MTTR', 'WIMI', 'VUZI', 'SNAP', 'FVRR', 'UPST', 'PATH', 'S', 'FANG', 'VRSN',

        # 바이오/유전자 편집
        'ILMN',
        'BEAM', 'NTLA', 'EDIT',

        # 네트워크/보안/클라우드
        'ERIC', 'NOK', 'INFN', 'LITE', 'FIVN', 'NTNX', 'PANW',

        # 드론/방산
        'UAVS', 'EH', 'KTOS', 'AVAV',

        # 양자 컴퓨팅
        'IONQ', 'RGTI', 'QBTS',

        # 데이터/소프트웨어
        'ESTC', 'DT', 'PD', 'AYX',
        'IRBT', 'RBOT', 'RWLK', 'ZBRA', 'ABB',

        # AI/소프트웨어
        'BBAI', 'SOUN', 'TER', 'ROK', 'FANUY', 'FARO',

        # [AI / 비전(Vision) / 챗봇 등]
        'LPSN', 'VERI', 'AMBA',

        # [AI 기반 신약개발 / 합성생물학]
        'RXRX', 'EXAI', 'DNA', 'TEAM', 'UNH',

        # [자율주행(로보틱스 응용)]
        'AUR', 'ETN', 'FTAI', 'JOBY', 'LULU', 'NXPI', 'NWSA', 'ON', 'ROKU', 'SEDG', 'SANA'
    ]

    PREDEFINED_SYMBOLS_SECOND = ['RXRX', 'TPG', 'PLMR', 'PINS', 'ZM', 'TMDX', 'PNS', 'UBER',
                                 'AVTR', 'RVLV', 'AKRO', 'CMBM', 'MIRM', 'NOVA', 'CSTL', 'DT', 'INMD', 'ALRS',
                                 'NVST', 'DDOG', 'MCBS', 'BNTX', 'BRBR', 'BWIN', 'PGNY', 'SITM', 'BILL', 'SPT',
                                 'VEL', 'ANVS', 'BDTX', 'ARQT', 'SDGR', 'BEAM', 'ONEW', 'NREF', 'RVMD', 'GFL', 'ELVN', 'KROS', 'NARI',
                                 'PLRX', 'LEGN', 'FOUR', 'RNA', 'AZEK', 'PCVX', 'RPRX', 'ACI', 'MEG', 'NRIX', 'VERX', 'LI', 'VITL', 'RKT',
                                 'IBEX', 'NTST', 'INBX', 'HRMY', 'KYMR', 'STEP', 'DYN', 'BNL', 'BSY', 'GLSI', 'YALA', 'ASAN', 'THRY', 'AVO', 'IMNM',
                                 'ASO', 'SQFT', 'ELUT', 'EBC', 'MNSO', 'PRAX', 'TARS', 'FHTX', 'MAX', 'ROOT', 'ALGM', 'SHC', 'DCBO',
                                 'PUBM', 'DASH', 'ABNB', 'CVRX', 'YOU', 'INTA', 'S', 'RYAN', 'USCB', 'DUOL', 'CWAN', 'NU',
                                 'DFH', 'IMCR', 'BVS', 'OSCR', 'SEMR', 'ALHC', 'KARO', 'COIN', 'APP', 'BMEA', 'RXRX', 'DV','EDR', 'BWMN', 'GLBE', 'VERA', 'FLYW', 'PAY',
                                 'DLO', 'ZETA', 'MNDY', 'TASK', 'JANX', 'FA', 'DOCS'
                                  ]

    # 2021  6월까지 확인완
    @staticmethod
    def get_stock_data(symbol, interval, limit):
        """
        symbol: 주식 심볼 (예: AAPL)
        interval: 데이터 간격 ('1d', '1wk', '1mo')
        limit: 조회할 데이터 수 (무시하고 period로 처리)
               AAPL: Period '1000d' is invalid, must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        """
        # interval에 따른 적절한 period 설정
        if interval == '1d':
            period = '6mo'  # 6개월
        elif interval == '1wk':
            period = 'max'  # 1년
        elif interval == '1mo':
            period = 'max'  # 2년
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        stock = yf.Ticker(symbol)

        try:
            df = stock.history(period=period, interval=interval)
            if df.empty:
                print(f"No data returned for symbol {symbol} with interval {interval} and period {period}.")
            return df
        except Exception as e:
            print(f"Error fetching data for symbol {symbol}: {str(e)}")
            return pd.DataFrame()  # 빈 데이터프레임 반환

    def calculate_indicators(self, data: Dict[str, List[float]]) -> pd.DataFrame:
        if not data or not data.get('close'):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        analyzer = UltimateRSIAnalyzer(df, length=9, smoType1='RMA', smoType2='EMA', smooth=7)
        df = analyzer.get_dataframe()
        indicator = SqueezeMomentumIndicator(df)
        df = indicator.get_dataframe()

        # NaN 처리
        df.fillna(0, inplace=True)  # 모든 NaN 값을 0으로 대체
        return df

    @staticmethod
    def format_stock_data(df):
        """
        데이터프레임을 특정 형식으로 변환
        """
        df.reset_index(inplace=True)
        df.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        return {
            'timestamp': df['timestamp'].astype(str).tolist(),
            'open': df['open'].tolist(),
            'high': df['high'].tolist(),
            'low': df['low'].tolist(),
            'close': df['close'].tolist(),
            'volume': df['volume'].tolist()
        }

    def get_symbol_status(self, symbol, intervals, limit):
        """
        심볼과 간격별 데이터를 확인하고 조건에 맞는지 여부를 반환합니다.
        """
        try:
            for label, interval in intervals.items():
                # 1d(일봉)은 건너뛰고 싶다면 아래에서 continue 처리
                if interval == "1d":
                    # 일봉은 아예 검사하지 않음
                    continue

                # 여기부터는 주봉(1wk) 또는 월봉(1mo)에 대해서만 조건을 적용
                df = self.get_stock_data(symbol, interval, limit)
                if df is None or df.empty:
                    return False  # 데이터 없음 -> 조건 불만족

                formatted_data = self.format_stock_data(df)
                df_with_indicators = self.calculate_indicators(formatted_data)

                if df_with_indicators.empty:
                    return False  # 지표 계산 실패 -> 조건 불만족

                last_row = df_with_indicators.iloc[-1].to_dict()

                # 주봉(1wk)과 월봉(1mo) 모두 동일한 조건 적용
                condition = (
                        last_row.get("RSI", 0) > last_row.get("RSI_signal", 0)
                        and last_row.get("SqueezeColor", "").lower() in {"lime", "maroon"}
                )

                if not condition:
                    return False  # 조건 불만족 시 즉시 종료

            return True  # 모든 간격(여기서는 주봉·월봉)에 대해 조건을 만족
        except Exception as e:
            print(f"Error processing symbol {symbol}: {str(e)}")
            return False  # 에러 발생 시 조건 불만족


    # def get_symbol_status(self, symbol, intervals, limit):
    #     """
    #     심볼과 간격별 데이터를 확인하고 조건에 맞는지 여부를 반환합니다.
    #     """
    #     try:
    #         for label, interval in intervals.items():
    #             df = self.get_stock_data(symbol, interval, limit)
    #             if df is None or df.empty:
    #                 return False  # 데이터 없음, 조건 불만족
    #
    #             formatted_data = self.format_stock_data(df)
    #             df_with_indicators = self.calculate_indicators(formatted_data)
    #
    #             if df_with_indicators.empty:
    #                 return False  # 지표 계산 실패, 조건 불만족
    #
    #             last_row = df_with_indicators.iloc[-1].to_dict()
    #
    #             # 각 간격별로 다른 조건 적용
    #             if interval in {"1d", "1wk"}:
    #                 # 일봉 및 주봉: RSI > RSI_signal AND SqueezeColor가 'lime' 또는 'maroon'
    #                 condition = (
    #                         last_row.get("RSI", 0) > last_row.get("RSI_signal", 0) and
    #                         last_row.get("SqueezeColor", "").lower() in {"lime", "maroon"}
    #                 )
    #             elif interval == "1mo":
    #                 # 월봉: SqueezeColor가 'lime' 또는 'maroon'만 확인
    #                 condition = last_row.get("SqueezeColor", "").lower() in {"lime", "maroon"}
    #             else:
    #                 # 지원되지 않는 간격
    #                 condition = False
    #
    #
    #
    #             if not condition:
    #                 return False  # 조건 불만족
    #
    #         return True  # 모든 간격에 대해 조건 만족
    #     except Exception as e:
    #         print(f"Error processing symbol {symbol}: {str(e)}")
    #         return False  # 에러 발생 시 조건 불만족

    def get_all_last_data(self, request, symbolList):
        limit = 500
        intervals = {
            '1day': '1d',
            '1week': '1wk',
            '1month': '1mo'
        }
        if symbolList == 1:
            all_symbols = self.PREDEFINED_SYMBOLS
        elif symbolList == 2:
            all_symbols = self.PREDEFINED_SYMBOLS_SECOND

        # all_symbols = self.PREDEFINED_SYMBOLS
        filtered_symbols = []

        try:
            # 병렬 처리 시작
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(self.get_symbol_status, sym, intervals, limit): sym
                    for sym in all_symbols
                }
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]
                    try:
                        if future.result():  # 조건 만족
                            filtered_symbols.append(symbol)
                    except Exception as e:
                        print(f"Error processing symbol {symbol}: {str(e)}")

            # 조건에 맞는 심볼 리스트 저장
            if filtered_symbols:
                StockData.objects.create(symbols=filtered_symbols)
                print(f'{len(filtered_symbols)} symbols saved to TradingRecord.')
            else:
                StockData.objects.create(symbols=[])
                print('No symbols met the criteria.')

            # 조건에 맞는 심볼 리스트만 반환
            return Response({"symbols_saved": filtered_symbols})
        except Exception as e:
            return Response(
                {'error': f'An unexpected error occurred: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    def get(self, request):
        all_last = request.GET.get('all_last', 'false').lower() == 'true'
        all_last_second = request.GET.get('all_last_second', 'false').lower() == 'true'

        if all_last:
            return self.get_all_last_data(request, 1)
        elif all_last_second:
            return self.get_all_last_data(request, 2)
        else:
            limit = 1000
            symbol = request.GET.get('symbol', 'AAPL')  # 기본 심볼: AAPL

            intervals = {
                '1day': '1d',
                '1week': '1wk',
                '1month': '1mo'  # 1개월 간격 추가
            }

            result_all_intervals = {}

            try:
                for label, api_interval in intervals.items():
                    # 데이터 가져오기
                    df = self.get_stock_data(symbol, api_interval, limit)
                    if df is None or df.empty:
                        result_all_intervals[label] = {'error': f'No data available for {api_interval}'}
                        continue

                    # 데이터 형식 변환
                    formatted_data = self.format_stock_data(df)
                    df_with_indicators = self.calculate_indicators(formatted_data)

                    # 최근 30개의 데이터 반환
                    records = df_with_indicators.iloc[-30:]

                    # NaN 처리
                    records = records.replace({float('nan'): None}).to_dict(orient='records')
                    result_all_intervals[label] = records

                return Response(result_all_intervals)

            except Exception as e:
                return Response(
                    {'error': f'An unexpected error occurred: {str(e)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )


# http://127.0.0.1:8000/api/v1/binanceData/KoreaStockData/?all_last=true
# http://127.0.0.1:8000/api/v1/binanceData/KoreaStockData/?symbol=005930
# https://gridtrade.one/api/v1/binanceData/KoreaStockData/?all_last=true
class KoreaStockDataView(APIView):
    # 한국 주식 심볼 (예: KRX 코드 사용)
    PREDEFINED_SYMBOLS = [
        # 대형주
        "005930",  # 삼성전자
        "000660",  # SK하이닉스
        "035420",  # NAVER
        "051910",  # LG화학
        "035720",  # 카카오
        "005380",  # 현대차
        "068270",  # 셀트리온
        "017670",  # SK텔레콤
        "028260",  # 삼성물산
        "105560",  # KB금융
        "096770",  # SK이노베이션
        "034730",  # SK
        "032830",  # 삼성생명
        "055550",  # 신한지주
        "003550",  # LG

        # 중소형주 (중목)
        "012330",  # 현대모비스
        "006400",  # 삼성SDI
        "034220",  # LG전자
        "051900",  # LG디스플레이
        "047810",  # 한국타이어
        "005490",  # POSCO
        "032640",  # 삼성화재
        "034020",  # S-Oil
        "035250",  # 한화솔루션
        "066970",  # 아모레퍼시픽
        "069620",  # KCC
        "018260",  # 미래에셋대우
        "086790",  # 하나금융지주
        "006360",  # 한국조선해양
        "091990",  # 한화에어로스페이스
        "015760",  # 한국공항
        "012510",  # 현대로템
        "011200",  # 두산중공업
        "033780",  # 삼성바이오로직스
        "003490",  # 대한항공
        "030200",  # 기아
        "011070",  # 삼성전자우
        "006800",  # 삼성에스디에스
        "009150",  # 삼성증권
        "010140",  # KOSPI100
        "068760",  # 하이브
        "000270",  # 현대해상
        "036570",  # 엔씨소프트
        "028050",  # 삼성카드
        "008770",  # GS건설
        "009540",  # LG유플러스
        "000250",  # 삼천당제약
        "053690",  # 한미글로벌

        #잡주
        "008930",  # 삼양사
        "002790",  # 현대바이오
        "004170",  # 포스코케미칼
        "078930",  # 메지온
        "069960",  # 지니언스
        "005940",  # 세방전지
        "011790",  # KC그룹
        "020150",  # 일진머티리얼즈
        "052690",  # 에스디생명공학
        "048260",  # 삼양옵틱스
        "095570",  # 아이에이치시스템
        "071050",  # 한국옵틱스
        "042670",  # 아이센스

        # 코스닥 종목 (KOSDAQ)
        "091170",  # 한화시스템
        "293490",  # 알테오젠
        "252670",  # 라온시큐어
        "207940",  # 씨젠
        "251270",  # 에코프로비엠
        "047050",  # 코아셈
        "089860",  # 에이프로젠제약
        "064350",  # 엘앤에프
        "095700",  # 한온시스템
        "038460",  # 솔브레인
        "053800",  # 한국지엠
        "122630",  # SK머티리얼즈
        "294870",  # 카카오게임즈
        "298690",  # 종근당
        "333430",
        "460930",
        "443060",
        "082740",
    ]

    @staticmethod
    def get_stock_data(symbol, interval, limit):
        """
        한국 주식 데이터를 pykrx를 통해 interval 별로 가져옵니다.
        symbol: KRX 종목 코드 (예: "005930" for 삼성전자)
        interval: 데이터 간격 ('1d', '1wk', '1mo')
        limit: 가져올 데이터의 최대 개수
        """
        try:
            today = pd.Timestamp.today()
            if interval == "1d":
                start_date = (today - pd.Timedelta(days=limit)).strftime("%Y%m%d")
            elif interval == "1wk":
                start_date = (today - pd.Timedelta(weeks=limit)).strftime("%Y%m%d")
            elif interval == "1mo":
                start_date = (today - pd.Timedelta(days=limit * 30)).strftime("%Y%m%d")
            else:
                raise ValueError(f"Unsupported interval: {interval}")

            end_date = today.strftime("%Y%m%d")
            df = stock.get_market_ohlcv_by_date(start_date, end_date, symbol)
            if df.empty:
                print(f"No data available for symbol {symbol}")
                return pd.DataFrame()

            # 리샘플링 처리
            df.index = pd.to_datetime(df.index)
            if interval == "1wk":
                df = df.resample('W').agg({
                    '시가': 'first',
                    '고가': 'max',
                    '저가': 'min',
                    '종가': 'last',
                    '거래량': 'sum'
                }).dropna()
            elif interval == "1mo":
                df = df.resample('ME').agg({
                    '시가': 'first',
                    '고가': 'max',
                    '저가': 'min',
                    '종가': 'last',
                    '거래량': 'sum'
                }).dropna()
            # '1d'는 이미 일별 데이터이므로 리샘플링 불필요

            return df
        except Exception as e:
            print(f"Error fetching data for symbol {symbol}: {str(e)}")
            return pd.DataFrame()  # 빈 데이터프레임 반환

    @staticmethod
    def format_stock_data(df):
        """
        pykrx 데이터프레임을 특정 형식으로 변환합니다.
        """
        df.reset_index(inplace=True)
        df.rename(columns={
            "날짜": "timestamp",
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
        }, inplace=True)
        return {
            "timestamp": df["timestamp"].astype(str).tolist(),
            "open": df["open"].tolist(),
            "high": df["high"].tolist(),
            "low": df["low"].tolist(),
            "close": df["close"].tolist(),
            "volume": df["volume"].tolist(),
        }

    def calculate_indicators(self, data: dict) -> pd.DataFrame:
        """
        RSI 및 Squeeze Momentum Indicator 계산 (미국 주식과 동일)
        """
        if not data or not data.get("close"):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        analyzer = UltimateRSIAnalyzer(df, length=9, smoType1="RMA", smoType2="EMA", smooth=7)
        df = analyzer.get_dataframe()
        indicator = SqueezeMomentumIndicator(df)
        df = indicator.get_dataframe()

        # NaN 처리
        df.fillna(0, inplace=True)
        return df

    def get_symbol_status(self, symbol, intervals, limit):
        """
        심볼과 간격별 데이터를 확인하고 조건에 맞는지 여부를 반환합니다.
        """
        try:
            for label, interval in intervals.items():
                # 1d(일봉)은 건너뛰고 싶다면 아래에서 continue 처리
                if interval == "1d":
                    # 일봉은 아예 검사하지 않음
                    continue

                # 여기부터는 주봉(1wk) 또는 월봉(1mo)에 대해서만 조건을 적용
                df = self.get_stock_data(symbol, interval, limit)
                if df is None or df.empty:
                    return False  # 데이터 없음 -> 조건 불만족

                formatted_data = self.format_stock_data(df)
                df_with_indicators = self.calculate_indicators(formatted_data)

                if df_with_indicators.empty:
                    return False  # 지표 계산 실패 -> 조건 불만족

                last_row = df_with_indicators.iloc[-1].to_dict()

                # 주봉(1wk)과 월봉(1mo) 모두 동일한 조건 적용
                condition = (
                        last_row.get("RSI", 0) > last_row.get("RSI_signal", 0)
                        and last_row.get("SqueezeColor", "").lower() in {"lime", "maroon"}
                )

                if not condition:
                    return False  # 조건 불만족 시 즉시 종료

            return True  # 모든 간격(여기서는 주봉·월봉)에 대해 조건을 만족
        except Exception as e:
            print(f"Error processing symbol {symbol}: {str(e)}")
            return False  # 에러 발생 시 조건 불만족


    # def get_symbol_status(self, symbol, intervals, limit):
    #     """
    #     심볼 데이터를 가져와 조건에 맞는지 확인합니다 (미국 주식과 동일한 조건).
    #     """
    #     try:
    #         for label, interval in intervals.items():
    #             df = self.get_stock_data(symbol, interval, limit)
    #             if df is None or df.empty:
    #                 return False  # 조건 불만족
    #
    #             formatted_data = self.format_stock_data(df)
    #             df_with_indicators = self.calculate_indicators(formatted_data)
    #
    #             if df_with_indicators.empty:
    #                 return False  # 지표 계산 실패
    #
    #             last_row = df_with_indicators.iloc[-1].to_dict()
    #
    #             # 조건 체크: RSI > RSI_signal 및 SqueezeColor가 lime 또는 maroon
    #             if not (last_row.get("RSI", 0) > last_row.get("RSI_signal", 0) and
    #                     last_row.get("SqueezeColor", "").lower() in {"lime", "maroon"}):
    #                 return False  # 조건 불만족
    #
    #         return True  # 모든 간격에 대해 조건 만족
    #     except Exception as e:
    #         print(f"Error processing symbol {symbol}: {str(e)}")
    #         return False  # 조건 불만족 또는 에러

    def get_all_last_data(self, request):
        limit = 500
        intervals = {
            "1day": "1d",
            "1week": "1wk",
            "1month": "1mo"
        }
        all_symbols = self.PREDEFINED_SYMBOLS
        filtered_symbols = []

        try:
            # 병렬 처리 시작
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(self.get_symbol_status, sym, intervals, limit): sym
                    for sym in all_symbols
                }
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]
                    try:
                        if future.result():  # 조건 만족
                            filtered_symbols.append(symbol)
                    except Exception as e:
                        print(f"Error processing symbol {symbol}: {str(e)}")


            # 조건에 맞는 심볼 리스트 저장
            if filtered_symbols:
                KoreaStockData.objects.create(symbols=filtered_symbols)
                print(f'{len(filtered_symbols)} symbols saved to TradingRecord.')
            else:
                KoreaStockData.objects.create(symbols=[])
                print('No symbols met the criteria.')


            # 조건에 맞는 심볼 리스트만 반환
            return Response({"symbols_saved": filtered_symbols})
        except Exception as e:
            return Response(
                {"error": f"An unexpected error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def get(self, request):
        """
        특정 심볼에 대해 interval 별로 데이터를 가져옵니다.
        """
        all_last = request.GET.get("all_last", "false").lower() == "true"

        if all_last:
            return self.get_all_last_data(request)
        else:
            limit = int(request.GET.get("limit", 500))  # 기본 limit 값
            symbol = request.GET.get("symbol", "005930")  # 기본 심볼: 삼성전자

            intervals = {
                "1day": "1d",
                "1week": "1wk",
                "1month": "1mo"
            }

            result_all_intervals = {}

            try:
                for label, api_interval in intervals.items():
                    # 데이터 가져오기
                    df = self.get_stock_data(symbol, api_interval, limit)
                    if df is None or df.empty:
                        result_all_intervals[label] = {"error": f"No data available for {api_interval}"}
                        continue

                    # 데이터 형식 변환
                    formatted_data = self.format_stock_data(df)
                    df_with_indicators = self.calculate_indicators(formatted_data)

                    if df_with_indicators.empty:
                        result_all_intervals[label] = {"error": "Indicator calculation failed"}
                        continue

                    # 최근 30개의 데이터 반환 (데이터가 부족할 경우 전체 데이터 반환)
                    records = df_with_indicators.iloc[-30:]
                    records = records.replace({pd.NA: None, float('nan'): None}).to_dict(orient="records")
                    result_all_intervals[label] = records

                return Response(result_all_intervals)

            except Exception as e:
                return Response(
                    {"error": f"An unexpected error occurred: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

class getCurrentPrice(APIView):
    def get(self, request):
        try:
            symbol = request.GET.get('symbol', 'BTCUSDT')  # 기본 심볼: BTCUSDT
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url)

            if response.status_code != 200:
                print(f"Error fetching current price: {response.status_code} - {response.text}")
                return Response({"error": "Failed to fetch data from Binance API"}, status=500)

            data = response.json()
            if 'price' not in data:
                print(f"Error: 'price' not found in response: {data}")
                return Response({"error": "'price' key not found in API response"}, status=500)

            # 성공적으로 가격을 가져온 경우
            return Response({"symbol": symbol, "price": float(data['price'])}, status=200)

        except Exception as e:
            print(f"Exception fetching current price for {symbol}: {e}")
            return Response({"error": str(e)}, status=500)







    # Yahoo Finance
    #
    # “숫자 종목코드 + .SS or .SZ”를 정확히 알고 있다면 yf.Ticker("600519.SS") 식으로 데이터를 가져올 수 있습니다.
    # 다만, 본토 A주 종목 데이터는 해외 주식 대비 제공 범위나 신뢰도(과거 데이터 범위 등)가 다를 수 있으며, 종종 업데이트가 지연될 수 있습니다.
    # Tushare (추천)
    #
    # 중국 로컬에서 많이 쓰는 무료/유료 API 서비스이며, Python 라이브러리 형태로 제공됨.
    # 회원가입 후 발급받은 토큰을 이용해 상하이·선전·홍콩 등 다양한 중국 증시 데이터를 받아올 수 있음.
    # 예시 코드:
    # python
    # 복사
    # import tushare as ts
    #
    # # 토큰 설정
    # ts.set_token('YOUR_TOKEN')
    # pro = ts.pro_api()
    #
    # # 일봉 데이터 가져오기 (예: 상하이 증시 600519 마오타이)
    # df = pro.daily(ts_code='600519.SH', start_date='20230101', end_date='20231231')
    # print(df.head())
    # 이런 식으로 숫자 종목코드 + .SH(상하이) / .SZ(선전) 형식의 ts_code를 사용합니다.


class ChinaStockDataView(APIView):
    """
    중국 본토/홍콩 상장 종목을 대상으로
    (1wk, 1mo) 조건( RSI > RSI_signal & SqueezeColor in {lime, maroon} )을 만족하는 심볼만 저장 및 반환
    """

    # 중국 본토/홍콩 주요 종목 예시
    # - 상하이(.SS) : 구이저우 마오타이(600519.SS), 핑안보험(601318.SS), 공상은행(601398.SS) 등
    # - 선전(.SZ) : BYD(002594.SZ), 메이디(000333.SZ), 우량예(000858.SZ) 등
    # - 홍콩(.HK) : 텐센트(0700.HK), 알리바바 HK(9988.HK), 메이퇀(3690.HK) 등
    CHINA_PREDEFINED_SYMBOLS = [
        # ───── AI / 로보틱스 / 드론 ─────
        "002415.SZ",  # Hikvision (하이크비전, AI 기반 CCTV/보안)
        "002230.SZ",  # iFlytek (아이플라이텍, 음성인식 AI)
        "0020.HK",  # SenseTime (센스타임, AI 비전)
        "000063.SZ",  # ZTE (중흥통신, 5G·통신장비, AI/네트워크 분야)
        "002747.SZ",  # Estun Automation (에스튼, 산업용 로봇)
        "300024.SZ",  # Siasun Robot & Automation (시아순, 중국 대표 로봇 기업)

        # AI 소형주
        "300223.SZ",  # Beijing THUNISOFT Corp (빅데이터 및 AI 플랫폼 개발)
        "300123.SZ",  # Sunyard Technology (AI 및 핀테크 소프트웨어 개발)
        "688100.SS",  # Hygon Information Technology (AI 칩 및 서버)
        "002253.SZ",  # Chinasoft International (AI 및 클라우드 솔루션 개발)

        # ───── 인터넷·소프트웨어·플랫폼 ─────
        "0700.HK",  # Tencent (텐센트, 게임·클라우드·AI)
        "1810.HK",  # Xiaomi (샤오미, IoT·스마트폰)
        "9988.HK",  # Alibaba (알리바바, 클라우드·AI·이커머스)
        "3690.HK",  # Meituan (메이퇀, 슈퍼앱·AI 배달)

        # ───── 전기차·자율주행·배터리 ─────
        "002594.SZ",  # BYD (비야디, EV·배터리)
        "1211.HK",  # BYD (홍콩 상장)
        "300750.SZ",  # CATL (닝더스다이, 배터리)
        "9868.HK",  # Xpeng (샤오펑, EV·자율주행)
        "2015.HK",  # Li Auto (리오토, EV·자율주행)

        # ───── 드론 / 항공 기술 ─────
        "002929.SZ",  # DJI (다장혁신, 드론 분야 글로벌 선두)
        "688011.SS",  # AVIC (중국항공산업그룹, 항공기 및 드론 제조)
        "002013.SZ",  # ShenZhen Protruly (프로트럴리, 드론 및 카메라 기술)

        # 드론 소형주
        "300159.SZ",  # New Dazheng Property Group (소형 드론 제조)
        "300489.SZ",  # TIANJIN TIANLONG TECHNOLOGY (산업용 드론 및 방제용 드론)
        "300728.SZ",  # Magnity Electronics (드론 부품 제조)
        "300679.SZ",  # Hangke Technology (드론 및 항공 기술 연구)

        # ───── 배터리 관련 ─────
        "002091.SZ",  # Meidu Energy (메이두에너지, 리튬 배터리 소재)
        "300014.SZ",  # Eve Energy (이브에너지, 리튬이온 배터리 제조)
        "002812.SZ",  # Yunnan Energy (윈난에너지, 리튬 배터리 핵심 소재)
        "600884.SS",  # Ganfeng Lithium (간펑리튬, 리튬 채굴 및 배터리)
        "002460.SZ",  # Huayou Cobalt (화유코발트, 배터리 원자재)

        # ───── 반도체 / AI 칩 ─────
        "688256.SS",  # Cambricon (캄브리콘, AI 칩)
        "0981.HK",  # SMIC (중국 반도체 파운드리, 홍콩)
        "1347.HK",  # Hua Hong Semiconductor (화훙반도체, 홍콩)

        # ───── 데이터 기술 / 빅데이터 ─────
        "000977.SZ",  # Inspur Electronic Information Industry (인스퍼, 서버 및 빅데이터 솔루션)
        "600588.SS",  # Yonyou Network Technology (용유네트워크, 기업용 소프트웨어 및 클라우드 서비스)
        "002065.SZ",  # Donghua Software (동화소프트웨어, 의료 및 금융 분야의 빅데이터 솔루션)
        "600271.SS",  # Anhui USTC iFlytek (아이플라이텍, 음성인식 및 AI 기술)
        "300002.SZ",  # Shenzhen Sunline Tech (썬라인테크, 금융 IT 솔루션 및 빅데이터)

        # ───── 게임 및 엔터테인먼트 ─────
        "0700.HK",  # Tencent (텐센트, 세계 최대 게임 퍼블리셔)
        "9999.HK",  # NetEase (넷이즈, MMORPG 및 모바일 게임 전문)
        "002624.SZ",  # Perfect World (퍼펙트월드, 온라인 RPG 전문 개발사)
        "300052.SZ",  # Zhejiang Century Huatong (절강화통, 게임·미디어 콘텐츠)
        "002555.SZ",  # Sanqi Interactive Entertainment (37 인터랙티브, 모바일 게임 중심)
        "600633.SS",  # CMGE (중국 대표 모바일 게임사)
        "300418.SZ",  # Alpha Group (알파그룹, 게임 및 애니메이션 콘텐츠)
    ]
    @staticmethod
    def get_stock_data(symbol, interval, limit):
        """
        symbol: 주식 심볼 (예: 600519.SS)
        interval: 데이터 간격 ('1d', '1wk', '1mo')
        limit: 조회할 데이터 수 (여기서는 period 매핑 시에만 사용)
        """
        # interval에 따른 적절한 period 설정
        if interval == '1d':
            period = '6mo'
        elif interval == '1wk':
            period = 'max'
        elif interval == '1mo':
            period = 'max'
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        stock = yf.Ticker(symbol)
        try:
            df = stock.history(period=period, interval=interval)
            if df.empty:
                print(f"No data returned for symbol {symbol} / interval {interval}")
            return df
        except Exception as e:
            print(f"Error fetching data for symbol {symbol}: {str(e)}")
            return pd.DataFrame()

    def calculate_indicators(self, data: Dict[str, List[float]]) -> pd.DataFrame:
        """
        'timestamp','open','high','low','close','volume' 형식의 데이터를
        지표 계산에 활용하는 예시
        """
        if not data or not data.get('close'):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        # RSI 지표 계산 (예시)
        analyzer = UltimateRSIAnalyzer(df, length=9)
        df = analyzer.get_dataframe()

        # SqueezeMomentumIndicator (예시)
        indicator = SqueezeMomentumIndicator(df)
        df = indicator.get_dataframe()

        # NaN 처리
        df.fillna(0, inplace=True)
        return df

    @staticmethod
    def format_stock_data(df: pd.DataFrame) -> Dict[str, List]:
        """
        야후 파이낸스에서 받아온 df를
        {'timestamp': [...], 'open': [...], ...} 형태의 dict로 변환
        """
        df.reset_index(inplace=True)
        df.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        return {
            'timestamp': df['timestamp'].astype(str).tolist(),
            'open': df['open'].tolist(),
            'high': df['high'].tolist(),
            'low': df['low'].tolist(),
            'close': df['close'].tolist(),
            'volume': df['volume'].tolist()
        }

    def get_symbol_status(self, symbol, intervals, limit):
        """
        심볼과 간격별 데이터를 확인하고
        주봉(1wk), 월봉(1mo) 모두 조건(RSI>RSI_signal & SqueezeColor in {lime, maroon})을 만족하는지 체크
        일봉(1d)은 건너뜀
        """
        try:
            for label, interval in intervals.items():
                # 일봉은 검사 제외
                if interval == "1d":
                    continue

                df = self.get_stock_data(symbol, interval, limit)
                if df.empty:
                    return False  # 데이터가 없으면 조건 불만족

                formatted_data = self.format_stock_data(df)
                df_with_indicators = self.calculate_indicators(formatted_data)
                if df_with_indicators.empty:
                    return False

                last_row = df_with_indicators.iloc[-1].to_dict()

                # 주봉·월봉 공통 조건
                condition = (
                    last_row.get("RSI", 0) > last_row.get("RSI_signal", 0)
                    and last_row.get("SqueezeColor", "").lower() in {"lime", "maroon"}
                )

                if not condition:
                    return False
            return True
        except Exception as e:
            print(f"Error processing symbol {symbol}: {str(e)}")
            return False

    def get_all_last_data(self, request):
        limit = 500
        intervals = {
            '1day': '1d',
            '1week': '1wk',
            '1month': '1mo'
        }
        all_symbols = self.CHINA_PREDEFINED_SYMBOLS
        filtered_symbols = []

        try:
            # 병렬 처리
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(self.get_symbol_status, sym, intervals, limit): sym
                    for sym in all_symbols
                }
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]
                    try:
                        if future.result():  # 조건 만족
                            filtered_symbols.append(symbol)
                    except Exception as e:
                        print(f"Error processing symbol {symbol}: {str(e)}")

            # DB 저장 (StockData에 'symbols'라는 필드가 있다고 가정)
            if filtered_symbols:
                ChinaStockData.objects.create(symbols=filtered_symbols)
                print(f'{len(filtered_symbols)} symbols saved to StockData.')
            else:
                ChinaStockData.objects.create(symbols=[])
                print('No symbols met the criteria.')

            return Response({"symbols_saved": filtered_symbols})
        except Exception as e:
            return Response(
                {'error': f'An unexpected error occurred: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def get(self, request):
        """
        all_last=true로 GET하면 주봉·월봉 조건에 부합하는 심볼을 필터링해서 반환
        아니면 단일 심볼에 대한 1d,1wk,1mo 데이터를 반환
        """
        all_last = request.GET.get('all_last', 'false').lower() == 'true'

        if all_last:
            return self.get_all_last_data(request)
        else:
            limit = 500
            symbol = request.GET.get('symbol', '600519.SS')  # 기본값: 구이저우 마오타이
            intervals = {
                '1day': '1d',
                '1week': '1wk',
                '1month': '1mo'
            }

            result_all_intervals = {}
            try:
                for label, api_interval in intervals.items():
                    df = self.get_stock_data(symbol, api_interval, limit)
                    if df.empty:
                        result_all_intervals[label] = {'error': f'No data for {api_interval}'}
                        continue

                    formatted_data = self.format_stock_data(df)
                    df_with_indicators = self.calculate_indicators(formatted_data)
                    records = df_with_indicators.iloc[-30:]  # 최근 30개
                    records = records.replace({float('nan'): None}).to_dict(orient='records')
                    result_all_intervals[label] = records

                return Response(result_all_intervals)

            except Exception as e:
                return Response(
                    {'error': f'An unexpected error occurred: {str(e)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )