import os
from crewai import Agent, Task, Crew, Process
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np
import re
from datetime import datetime
from django.conf import settings
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
from TradeStrategy.models import StrategyConfig

import json
from decimal import Decimal

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

def analyze_with_gpt4(market_data, trendline_prices_str, current_status, currentPrice, last_decisions):
    try:
        # OpenAI API 키 설정

        client = OpenAI(
            api_key=settings.OPENAI_API_KEY,  # This is the default and can be omitted
        )
        # 지침 파일 읽기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        instructions_path = os.path.join(current_dir, 'instructions_v1.md')

        with open(instructions_path, 'r', encoding='utf-8') as file:
            instructions = file.read()

        # 대화 메시지 구성
        messages = [
            {"role": "user", "content": instructions},
            {"role": "user", "content": json.dumps(market_data)},
            {"role": "user", "content": json.dumps(trendline_prices_str)},  # 수정된 부분
            {"role": "user", "content": json.dumps(current_status)},
            {"role": "user", "content": str(currentPrice)},  # 문자열로 변환
            {"role": "user", "content": last_decisions},
        ]

        # OpenAI API 호출
        # response = client.chat.completions.create(
        #     model="o1-mini",  # 올바른 모델 이
        #     messages=messages,
        #     response_format={"type": "json_object"},
        #
        # )
        response = client.chat.completions.create(
            model="o3-mini",  # 올바른 모델 이름
            messages=messages
        )
        # 응답 내용 추출
        print("Raw API Response:", response)


        raw_content = response.choices[0].message.content


        # 코드 블록(````json`) 제거
        if raw_content.startswith("```json") and raw_content.endswith("```"):
            raw_content = raw_content[7:-3].strip()

        # JSON 파싱
        try:
            result = json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONDecodeError: {e}, 응답 내용: {raw_content}")


        grid_strategy = "RegularGrid"
        # 결정에 따른 grid_strategy 설정
        decision = result.get("decision", "HOLD").upper()

        if decision in ("BUY", "LONGGRID"):
            grid_strategy = "LongGrid"
        elif decision in ("SELL", "SHORTGRID"):
            grid_strategy = "ShortGrid"
        else:
            grid_strategy = "RegularGrid"


        validated_result = {
            "decision": result.get("decision", "HOLD").upper(),
            "percentage": min(max(float(result.get("percentage", 0)), 0), 100),  # 0-100 사이로 제한
            "reason": str(result.get("reason", "No reason provided")),
            "grid_strategy": grid_strategy,  # 추가된 부분
            "Multiple": float(result.get("Multiple", 0))  # max() 함수 제거
        }

        if validated_result["decision"] not in ["BUY", "SELL", "HOLD"]:
            validated_result["decision"] = "HOLD"

        return validated_result

    except Exception as e:
        return {
            "decision": "HOLD",
            "percentage": 0,
            "reason": f"Analysis failed: {str(e)}",
            "grid_strategy": "RegularGrid",  # 오류 발생 시 기본 전략 설정
            "Multiple": 0
        }


# def analyze_with_gpt4(market_data, trendline_prices_str, current_status, currentPrice, last_decisions):
#     try:
#             # 현재 파일의 디렉토리 경로를 가져옴
#         import os
#
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         instructions_path = os.path.join(current_dir, 'instructions_v1.md')
#
#         with open(instructions_path, 'r', encoding='utf-8') as file:
#                 instructions = file.read()
#
#         messages = [
#                 {"role": "system", "content": instructions},
#                 {"role": "user", "content": json.dumps(market_data)},
#                 {"role": "user", "content": trendline_prices_str},
#                 {"role": "user", "content": current_status},
#                 {"role": "user", "content": currentPrice},
#                 {"role": "user", "content": last_decisions},
#
#         ]
#         openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
#             # model = "o1-mini",  # 이미지를 처리할 수 있는 모델로 변경
#
#         response = openai_client.chat.completions.create(
#                 model="gpt-4o",  # 이미지를 처리할 수 있는 모델로 변경
#                 messages=messages,
#                 response_format={"type": "json_object"},
#                 max_tokens=800
#         )
#
#
#         result = json.loads(response.choices[0].message.content)
#         print(str(result))
#         print("----------------------------")
#             # 응답 검증 및 기본값 설정
#         if not isinstance(result, dict):
#             raise ValueError("GPT response is not a dictionary")
#
#         # 필수 필드 검증 및 기본값 설정
#         validated_result = {
#                 "decision": result.get("decision", "HOLD").upper(),
#                 "percentage": min(max(float(result.get("percentage", 0)), 0), 100),  # 0-100 사이로 제한
#                 "reason": str(result.get("reason", "No reason provided"))
#                 }
#
#         if validated_result["decision"] not in ["BUY", "SELL", "HOLD"]:
#                 validated_result["decision"] = "HOLD"
#
#         return validated_result
#
#     except Exception as e:
#         return {
#                 "decision": "HOLD",
#                 "percentage": 0,
#                 "reason": f"Analysis failed: {str(e)}"
#         }

def update_strategy_config():
    try:
        # StrategyConfig 모델에서 설정 찾기
        strategy_config = StrategyConfig.objects.get(name='240824')

        # 현재 config 가져오기
        current_config = strategy_config.config

        # 각 Symbol별로 최신 분석 결과 가져오기
        symbols = ['BTCUSDT', 'ETHUSDT']
        for symbol in symbols:
            latest_analysis = Analysis.objects.filter(symbol=symbol).order_by('-created_at').first()

            if latest_analysis:
                selected_strategy = latest_analysis.selected_strategy

                # Symbol에 따라 적절한 설정 업데이트
                if symbol == 'BTCUSDT':
                    if 'INIT' in current_config and 'setting' in current_config['INIT']:
                        current_config['INIT']['setting']['grid_strategy'] = selected_strategy
                elif symbol == 'ETHUSDT':
                    if 'ETH' in current_config and 'setting' in current_config['ETH']:
                        current_config['ETH']['setting']['grid_strategy'] = selected_strategy

                print(f"Updated StrategyConfig grid_strategy for {symbol} to {selected_strategy}")
            else:
                print(f"No AnalysisResult found for {symbol}")

        # 업데이트된 config 저장
        strategy_config.config = current_config
        strategy_config.save()

    except StrategyConfig.DoesNotExist:
        print("StrategyConfig with name '240824' not found")
    except Exception as e:
        print(f"Error updating StrategyConfig: {str(e)}")





def perform_new_analysis():
    try:
        config = get_strategy_config()
        if not config:
            print("Strategy configuration is invalid.")
            return None

        vt_symbol = config['vt_symbol']
        grid_strategy = config['grid_strategy']

        print(f"Current grid_strategy: {grid_strategy}")

        # Fetch Bitcoin data
        bitcoin_data = get_bitcoin_data_from_api(vt_symbol)

        intervals = ['15m', '1h', '2h', '1d']

        trendline_prices = get_trendlines_prices(vt_symbol, intervals)
        prices_15m = get_trendline_prices_for_interval(trendline_prices, '15m')
        prices_1h = get_trendline_prices_for_interval(trendline_prices, '1h')
        prices_2h = get_trendline_prices_for_interval(trendline_prices, '2h')
        prices_1d = get_trendline_prices_for_interval(trendline_prices, '1d')
        # prices_1w = get_trendline_prices_for_interval(trendline_prices, '1w')

        # {"1h": {"RecentSteepHigh": ["98035.43", "99608.75", "96584.75", "98035.43"],
        #         "RecentSteepLow": ["92260.19", "102015.30", "104162.20", "96036.08", "268604.69"],
        #         "LongTermHigh": ["98035.43", "98035.43"], "LongTermLow": ["96744.57"]},
        #  "2h": {"RecentSteepHigh": ["99608.68", "99608.68"],
        #         "RecentSteepLow": ["136709.74", "81270.92", "98732.73", "103538.72", "84359.02"],
        #         "LongTermHigh": ["99608.68", "99608.68"], "LongTermLow": ["69562.46"]},
        #  "1d": {"RecentSteepHigh": [], "RecentSteepLow": ["64279.73", "75994.12", "146494.73", "69755.92", "74277.73"],
        #         "LongTermHigh": [], "LongTermLow": ["51528.19"]},
        #  "1w": {"RecentSteepHigh": [], "RecentSteepLow": ["39863.44", "44985.84", "56411.49", "56592.99", "50417.25"],
        #         "LongTermHigh": [], "LongTermLow": ["20552.50"]}}
        # Convert trendline prices to JSON string
        trendline_prices_str = json.dumps({
            '15m': {k: [f"{price:.2f}" for price in v] for k, v in prices_15m.items()},
            '1h': {k: [f"{price:.2f}" for price in v] for k, v in prices_1h.items()},
            '2h': {k: [f"{price:.2f}" for price in v] for k, v in prices_2h.items()},
            '1d': {k: [f"{price:.2f}" for price in v] for k, v in prices_1d.items()},
        })
        # '1w': {k: [f"{price:.2f}" for price in v] for k, v in prices_1w.items()}

        print(trendline_prices_str)

        if not bitcoin_data:
            print("Failed to fetch bitcoin data.")
            return None

        # Fetch account balances and positions
        futures_usdt_balance = get_future_account("get-future-balance")
        available_balance = futures_usdt_balance.get("availableBalance", 0)

        futures_positions = get_future_account("get-future-position")
        filtered_positions = [position for position in futures_positions if position["symbol"] == vt_symbol]

        # 필터링 결과가 비어 있다면 코인 밸런스를 0으로
        if len(filtered_positions) > 0:
            coin_balance = Decimal(filtered_positions[0]["positionAmt"])
        else:
            coin_balance = Decimal("0")


        current_status = {
            "availableBalance": available_balance,
            "positions": filtered_positions
        }
        current_status_json = json.dumps(current_status, ensure_ascii=False, indent=4)

        current_price_data = get_current_price(symbol)
        # {'symbol': 'BTCUSDT', 'price': 98708.49}

        print(current_price_data)
        current_price = current_price_data.get('price') if current_price_data else 0

        last_decisions = get_last_decisions(current_price if current_price else 100000000)

        # GPT-4 analysis
        decision = analyze_with_gpt4(
            bitcoin_data,
            trendline_prices_str,
            current_status_json,
            current_price,
            last_decisions,
        )
        if not decision:
            print("Decision analysis failed.")
            return None


        grid_strategy = decision['grid_strategy']

        # [{'symbol': 'BTCUSDT', 'positionAmt': '0.008', 'entryPrice': '94501.4', 'breakEvenPrice': '94548.6507',
        #   'markPrice': '94520.10000000', 'unRealizedProfit': '0.14960000', 'liquidationPrice': '90089.53755020',
        #   'leverage': '20', 'maxNotionalValue': '100000000', 'marginType': 'isolated', 'isolatedMargin': '38.32736480',
        #   'isAutoAddMargin': 'false', 'positionSide': 'BOTH', 'notional': '756.16080000',
        #   'isolatedWallet': '38.17776480', 'updateTime': 1735017608142, 'isolated': True, 'adlQuantile': 2,
        #   'profit_percentage': 0.4}]

        # Create trading record
        trading_record = Analysis.objects.create(
            symbol=vt_symbol,
            trade_type=decision['decision'].upper(),
            result_string=decision['reason'],
            balance=Decimal(available_balance),
            coin_balance= coin_balance,
            current_price=Decimal(current_price),
            selected_strategy=grid_strategy,
            price_prediction=decision['Multiple']
        )
        update_strategy_config()
        return f"Analysis completed successfully in seconds. AnalysisResult id: {trading_record.id}"
    except Exception as e:
        print(f"Analysis error: {e}")
        return None