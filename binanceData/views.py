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
from typing import List, Dict, Optional
import bisect
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from functools import lru_cache
from django.core.cache import caches
import json
from pandas import Timestamp
from datetime import datetime, timezone


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
                '2h': 540,  # �� 41일
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
            fifteen_min_candles = self.get_extended_kline_data(symbol, '30m')
            thirty_min_candles = self.get_extended_kline_data(symbol, '1h')
            hourly_candles = self.get_extended_kline_data(symbol, '2h')
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
                # df["STD"] = df["close"].rolling(window=period).std()
                # df["Upper"] = df["MA"] + (df["STD"] * multiplier)
                # df["Lower"] = df["MA"] - (df["STD"] * multiplier)

                # MACD
                # exp1 = df['close'].ewm(span=12, adjust=False).mean()
                # exp2 = df['close'].ewm(span=26, adjust=False).mean()
                # df['MACD'] = exp1 - exp2
                # df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

                StochasticRSI(df)
                RSIAnalyzer(df)
                # 시간 프레임별 일목균형표 설정값 정의
                if timeframe == '15min':
                    ichimoku_settings = {'tenkan': 7, 'kijun': 22, 'senkou_b': 44, 'displacement': 22}
                elif timeframe == '30min':
                    ichimoku_settings = {'tenkan': 9, 'kijun': 26, 'senkou_b': 52, 'displacement': 26}
                elif timeframe == '2hourly':
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
                '15min': process_candles(fifteen_min_candles, '30min'),
                '30min': process_candles(thirty_min_candles, '1hourly'),
                'hourly': process_candles(hourly_candles, '2hourly'),
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
