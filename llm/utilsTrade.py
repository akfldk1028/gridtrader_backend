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
from .models import Analysis
from openai import OpenAI
import json

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
            data = response.json()  # JSON 응답 데이터 파싱
            return data
        except requests.exceptions.RequestException as e:
            print(f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"최대 재시도 횟수 초과. API 호출 실패: {base_url}")
                raise

    return None  # 이 줄은 실행되지 않지만, 함수의 모든 경로에서 반환값이 있음을 보장합니다.


def get_last_decisions(self, num_decisions: int = 3, current_price = 100000000) -> str:
        """Fetch recent trading decisions from database"""
        try:
            decisions = Analysis.objects.order_by('-created_at')[:num_decisions]

            formatted_decisions = []
            for decision in decisions:
                formatted_decision = {
                    "timestamp": int(decision.created_at.timestamp() * 1000),
                    "decision": decision.selected_strategy.lower(),
                    "reason": decision.result_string,
                    "balance": float(decision.balance),
                    "coin_balance": float(decision.coin_balance),
                    "current_price": {current_price},
                    "avg_buy_price": float(decision.avg_buy_price)  # btc_avg_buy_price -> avg_buy_price로 수정
                }
                formatted_decisions.append(str(formatted_decision))

            return "\n".join(formatted_decisions)
        except Exception as e:
            return ""


# TODO 이전의 RESULT STRING 값들 가져와서 추론 하기

def analyze_with_gpt4(market_data, trendline_prices_str, available_Balance, filtered_positions, currentPrice, last_decisions):
    try:
            # 현재 파일의 디렉토리 경로를 가져옴
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        instructions_path = os.path.join(current_dir, 'instructions_v3.md')

        with open(instructions_path, 'r', encoding='utf-8') as file:
                instructions = file.read()

        messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(market_data)},
                {"role": "user", "content": trendline_prices_str},
                {"role": "user", "content": trendline_prices_str},

        ]
        openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

        response = openai_client.chat.completions.create(
                model="gpt-4o",  # 이미지를 처리할 수 있는 모델로 변경
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=800
        )


        result = json.loads(response.choices[0].message.content)
        print(str(result))
        print("----------------------------")
            # 응답 검증 및 기본값 설정
        if not isinstance(result, dict):
            raise ValueError("GPT response is not a dictionary")

        # 필수 필드 검증 및 기본값 설정
        validated_result = {
                "decision": result.get("decision", "HOLD").upper(),
                "percentage": min(max(float(result.get("percentage", 0)), 0), 100),  # 0-100 사이로 제한
                "reason": str(result.get("reason", "No reason provided"))
                }

        if validated_result["decision"] not in ["BUY", "SELL", "HOLD"]:
                validated_result["decision"] = "HOLD"

        return validated_result

    except Exception as e:
        return {
                "decision": "HOLD",
                "percentage": 0,
                "reason": f"Analysis failed: {str(e)}"
        }


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
    # print(available_Balance) 30.36172625
    futures_positions = get_future_account("get-future-position")
    filtered_positions = [position for position in futures_positions if position["symbol"] == vt_symbol]
    # print(filtered_positions) []
    currentPrice = get_current_price(symbol)

    # {
    #     "symbol": "BTCUSDT",
    #     "price": 101083.64
    # }
    print(trendline_prices_str)
    last_decisions = get_last_decisions(currentPrice)

    # GPT-4 분석
    # decision = analyze_with_gpt4(
    #     bitcoin_data,
    #     trendline_prices_str,
    #     available_Balance,
    #     filtered_positions,
    #     currentPrice,
    #     last_decisions,
    # )

