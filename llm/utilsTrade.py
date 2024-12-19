import os
from crewai import Agent, Task, Crew, Process
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np
import re
from datetime import datetime
from django.conf import settings
from TradeStrategy.models import StrategyConfig
# from common.models import CommonModel
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from .analysis.StochasticRSI import StochasticRSI
from .analysis.rsi import RSIAnalyzer
from datetime import datetime, timedelta
import requests
from rest_framework.test import APIRequestFactory
from binanceData.views import BinanceLLMChartDataAPIView
from rest_framework.response import Response

# Binance 클라이언트 초기화
symbol = "BTCUSDT"

# binance_api_key = settings.BINANCE_API_KEY
# binance_api_secret = settings.BINANCE_API_SECRET
# client = Client(binance_api_key, binance_api_secret)


def extract_prediction(selected_strategy, result_string):
    # Determine direction based on selected_strategy
    if 'shortgrid' in selected_strategy.lower():
        direction = 'Down'
    elif 'longgrid' in selected_strategy.lower():
        direction = 'Up'
    elif 'regulargrid' in selected_strategy.lower():
        direction = 'Normal'
    else:
        direction = None

    # Extract percentage from result_string
    percentage_match = re.search(r'(\d+(?:\.\d+)?)%', result_string, re.IGNORECASE)

    if percentage_match:
        percentage = percentage_match.group(1)
    else:
        percentage = "0"

    return direction, percentage


def extract_strategy(text):
    strategies = ["LongGrid", "ShortGrid", "RegularGrid"]
    lines = text.split('\n')

    # 끝에서부터 각 줄을 검토
    for line in reversed(lines):
        for strategy in strategies:
            if strategy in line:
                return strategy

    # 아무 전략도 찾지 못했을 경우 RegularGrid 반환
    return "RegularGrid"


# def get_current_bitcoin_price(vt_symbol):
#     try:
#         ticker = client.get_symbol_ticker(symbol=vt_symbol)
#         return float(ticker['price'])
#     except BinanceAPIException as e:
#         print(f"Error fetching current Bitcoin price: {e}")
#         return None
def get_current_bitcoin_price(vt_symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={vt_symbol}"
        response = requests.get(url)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        print(f"Error fetching current Bitcoin price: {e}")
        return None


def get_strategy_config(strategy_name='240824'):
    try:
        strategy_config = StrategyConfig.objects.get(name=strategy_name)
        first_config = strategy_config.config.get('INIT', {})
        vt_symbol = first_config.get('vt_symbol', '')
        symbol = vt_symbol.split('.')[0] if vt_symbol else ''
        grid_strategy = first_config.get('setting', {}).get('grid_strategy')

        if not symbol or not grid_strategy:
            raise ValueError("Invalid configuration: missing symbol or grid_strategy")

        return {
            'vt_symbol': symbol,
            'grid_strategy': grid_strategy
        }
    except ObjectDoesNotExist:
        print(f"Strategy configuration not found for: {strategy_name}")
    except ValueError as e:
        print(f"Invalid configuration for {strategy_name}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in get_strategy_config: {str(e)}")


def get_bitcoin_data_from_api(symbol, max_retries=5, retry_delay=1):
    import time
    base_url = "https://gridtrade.one/api/v1/binanceData/llm-bitcoin-data/?all_last=false"
    params = {'symbol': symbol}

    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"최대 재시도 횟수 초과. API 호출 실패: {base_url}")
                raise

    return None  # 이 줄은 실행되지 않지만, 함수의 모든 경로에서 반환값이 있음을 보장합니다.


def get_trendlines_data(symbol, interval, max_retries=5, retry_delay=1):
    base_url = "https://gridtrade.one/api/v1/binanceData/trendLines/"
    full_url = f"{base_url}{symbol}/{interval}/"
    import time

    for attempt in range(max_retries):
        try:
            response = requests.get(full_url, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"트렌드라인 API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"최대 재시도 횟수 초과. 트렌드라인 API 호출 실패: {full_url}")
                raise

    return None

def extract_current_prices(trendlines_data):
    current_prices = {}
    for category, lines in trendlines_data['trend_lines'].items():
        current_prices[category] = [line['CurrentPrice'] for line in lines]
    return current_prices


def get_trendlines_prices(symbol, intervals):
    results = {}
    for interval in intervals:
        try:
            trendlines_data = get_trendlines_data(symbol, interval)
            if trendlines_data:
                results[interval] = extract_current_prices(trendlines_data)
        except Exception as e:
            print(f"Error fetching data for {symbol} at {interval}: {e}")
            results[interval] = None
    return results

def get_trendline_prices_for_interval(trendline_prices, interval):
    if interval in trendline_prices:
        return trendline_prices[interval]
    else:
        print(f"No data available for interval: {interval}")
        return None

def get_future_account(viewName, max_retries=5, retry_delay=1):
    import time

    base_url = f"https://gridtrade.one/api/v1/binanaceAccount/{viewName}"


    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, timeout=60)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"최대 재시도 횟수 초과. API 호출 실패: {base_url}")
                raise

    return None  # 이 줄은 실행되지 않지만, 함수의 모든 경로에서 반환값이 있음을 보장합니다.

def get_current_price(symbol, max_retries=5, retry_delay=1):
    import time

    base_url = "https://gridtrade.one/api/v1/binanceData/currentPrice/"
    params = {'symbol': symbol}

    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
            return response
        except requests.exceptions.RequestException as e:
            print(f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"최대 재시도 횟수 초과. API 호출 실패: {base_url}")
                raise

    return None  # 이 줄은 실행되지 않지만, 함수의 모든 경로에서 반환값이 있음을 보장합니다.


def get_last_decisions(self, num_decisions: int = 5, current_price = 100000000) -> str:
        """Fetch recent trading decisions from database"""
        try:
            # decisions = TradingRecord.objects.order_by('-created_at')[:num_decisions]

            formatted_decisions = []
            for decision in formatted_decisions:
                formatted_decision = {
                    "timestamp": int(decision.created_at.timestamp() * 1000),
                    "decision": decision.trade_type.lower(),
                    "percentage": float(decision.trade_ratio),
                    "reason": decision.trade_reason,
                    "btc_balance": float(decision.coin_balance),
                    "krw_balance": float(decision.balance),
                    "current_price": {current_price},
                    "avg_buy_price": float(decision.avg_buy_price)  # btc_avg_buy_price -> avg_buy_price로 수정
                }
                formatted_decisions.append(str(formatted_decision))

            return "\n".join(formatted_decisions)
        except Exception as e:
            return ""


# TODO 이전의 RESULT STRING 값들 가져와서 추론 하기
# 기존 get_current_bitcoin_price 함수 내용...

def  perform_new_analysis():
    config = get_strategy_config()
    if not config:
        print("Strategy configuration is invalid.")
        return None

    vt_symbol = config['vt_symbol']
    grid_strategy = config['grid_strategy']

    print(f"Current grid_strategy: {grid_strategy}")

    # bitcoin_data = get_bitcoin_data(vt_symbol)
    bitcoin_data = get_bitcoin_data_from_api(vt_symbol)
    intervals = ['1h', '2h','1d', '1w']

    trendline_prices = get_trendlines_prices(vt_symbol, intervals)
    prices_15m = get_trendline_prices_for_interval(trendline_prices, '1h')
    prices_30m = get_trendline_prices_for_interval(trendline_prices, '2h')
    prices_2h = get_trendline_prices_for_interval(trendline_prices, '1d')
    prices_6h = get_trendline_prices_for_interval(trendline_prices, '1w')
    # 트렌드 라인 가격 데이터를 문자열로 변환
    trendline_prices_str = {
        '1h': {k: [f"{price:.2f}" for price in v] for k, v in prices_15m.items()},
        '2h': {k: [f"{price:.2f}" for price in v] for k, v in prices_30m.items()},
        '1d': {k: [f"{price:.2f}" for price in v] for k, v in prices_2h.items()},
        '1w': {k: [f"{price:.2f}" for price in v] for k, v in prices_6h.items()}
    }



    if not bitcoin_data:
        print("Failed to fetch bitcoin data.")
        return None

    futures_usdt_balance = get_future_account("get-future-balance")
    available_Balance = futures_usdt_balance["availableBalance"]
    print(available_Balance)
    futures_positions = get_future_account("get-future-position")
    filtered_positions = [position for position in futures_positions if position["symbol"] == vt_symbol]
    print(filtered_positions)
    currentPrice = get_current_price(symbol)

    # {
    #     "symbol": "BTCUSDT",
    #     "price": 101083.64
    # }
    print(currentPrice)
    return  "0"
    # last_decisions = get_last_decisions(current_price)

    # GPT-4 분석
    # decision = analyzer.analyze_with_gpt4(
    #     bitcoin_data,
    #     last_decisions,
    #     available_Balance,
    #     filtered_positions
    # )
    #
    #
    #
    # trading_record = TradingRecord.objects.create(
    #     exchange='UPBIT',
    #     coin_symbol=symbol.split('-')[1],
    #     trade_type=decision['decision'].upper(),
    #     trade_ratio=Decimal(str(decision['percentage'])),
    #     trade_reason=decision['reason'],
    #     coin_balance=Decimal(current_status_dict[f'{symbol.split("-")[1].lower()}_balance']),
    #     balance=Decimal(current_status_dict['krw_balance']),
    #     current_price=Decimal(str(current_price)),
    #     trade_reflection=reflection,
    #     avg_buy_price=Decimal(str(avg_buy_price)),  # btc_avg_buy_price -> avg_buy_price로 수정
    # )


# last_decisions
