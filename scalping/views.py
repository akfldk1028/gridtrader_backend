# scalping/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.conf import settings
import websocket
import json
import threading
import os
from .models import TradingRecord
from decimal import Decimal
from datetime import datetime
# scalping/views.py
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import time


@method_decorator(csrf_exempt, name='dispatch')
class WebSocketTestView(View):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ws = None
        self.current_response = []
        self.is_connected = False
        self.response_received = threading.Event()

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            print(f"Received message: {data}")

            if data.get('type') == 'session.created':
                self.is_connected = True
                self.create_response(ws)
                return

            # 메시지 내용 처리
            if data.get('type') == 'conversation.item.message.content.part':
                content = data.get('content', '')
                if content:
                    self.current_response.append(content)
                    print(f"Received content: {content}")
                return

            # 응답 완료 처리
            if data.get('type') == 'response.done':
                if data.get('response', {}).get('status') == 'completed':
                    self.response_received.set()

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")

    def create_response(self, ws):
        """초기 응답 생성 요청"""
        response_message = {
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "instructions": "You are a trading assistant. Analyze BTC/USD price action and provide concise analysis."
            }
        }
        ws.send(json.dumps(response_message))

    def send_analysis_request(self, ws):
        """분석 요청 전송"""
        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",  # 'text' 대신 'input_text' 사용
                    "text": "Please analyze current BTC/USD price action and key levels."
                }]
            }
        }
        ws.send(json.dumps(message))

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        self.is_connected = False
        self.response_received.set()

    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        self.response_received.set()

    def on_open(self, ws):
        print("WebSocket connected")
        self.is_connected = True

    def post(self, request):
        try:
            if not self.ws or not self.is_connected:
                websocket.enableTrace(True)
                self.ws = websocket.WebSocketApp(
                    "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
                    header={
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                        "OpenAI-Beta": "realtime=v1"
                    },
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )

                wst = threading.Thread(target=self.ws.run_forever)
                wst.daemon = True
                wst.start()

                time.sleep(2)  # 연결 대기

            if not self.is_connected:
                return JsonResponse({
                    'status': 'error',
                    'message': 'WebSocket not connected'
                }, status=503)

            # 응답 추적 초기화
            self.current_response = []
            self.response_received.clear()

            # 분석 요청 전송 전에 잠시 대기
            time.sleep(1)

            # 분석 요청 전송
            self.send_analysis_request(self.ws)

            # 응답 대기 (최대 30초)
            if self.response_received.wait(timeout=30):
                response_text = ' '.join(self.current_response)
                return JsonResponse({
                    'status': 'success',
                    'response': response_text if response_text else 'No content received'
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Response timeout'
                }, status=504)

        except Exception as e:
            print(f"Exception occurred: {e}")
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)

    def get(self, request):
        return JsonResponse({
            'status': 'connected' if self.is_connected else 'disconnected',
            'response': ' '.join(self.current_response) if self.current_response else None
        })

    def delete(self, request):
        if self.ws:
            self.ws.close()
            self.ws = None
            self.current_response = []
            self.is_connected = False
        return JsonResponse({
            'status': 'success',
            'message': 'WebSocket connection closed'
        })