from channels.generic.websocket import AsyncWebsocketConsumer
import json
import websockets
import asyncio
from django.conf import settings
import aiohttp
from datetime import datetime


class OpenAIConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.openai_ws = None
        self.analysis_task = None
        await self.accept()
        await self.initialize_openai_connection()
        self.analysis_task = asyncio.create_task(self.periodic_bitcoin_analysis())

    async def initialize_openai_connection(self):
        try:
            url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }

            self.openai_ws = await websockets.connect(url, extra_headers=headers)

            # 더 간단한 지시사항으로 변경
            await self.openai_ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "instructions": """
                    비트코인 가격을 보고 매수/매도/홀드 중 하나를 추천해주세요.
                    답변은 반드시 다음 형식으로만 해주세요:
                    추천: [매수/매도/홀드]
                    이유: [한 문장으로 설명]
                    """
                }
            }))

            asyncio.create_task(self.forward_openai_messages())

        except Exception as e:
            await self.send(json.dumps({
                "type": "error",
                "message": f"OpenAI 연결 실패: {str(e)}"
            }))
            await self.close()

    async def get_bitcoin_data(self):
        """간단한 비트코인 가격 데이터만 가져오기"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true') as response:
                    data = await response.json()
                    return {
                        'price': data['bitcoin']['usd'],
                        'change_24h': data['bitcoin']['usd_24h_change']
                    }
        except Exception as e:
            return None

    async def periodic_bitcoin_analysis(self):
        """1분마다 간단한 분석 수행"""
        while True:
            try:
                if self.openai_ws and self.openai_ws.open:
                    bitcoin_data = await self.get_bitcoin_data()

                    if bitcoin_data:
                        analysis_request = {
                            "type": "message",
                            "content": f"""
                            현재 비트코인:
                            가격: ${bitcoin_data['price']:,.2f}
                            24시간 변동: {bitcoin_data['change_24h']:.2f}%

                            위 데이터를 보고 매수/매도/홀드 중 하나를 추천해주세요.
                            """
                        }

                        await self.openai_ws.send(json.dumps(analysis_request))

                await asyncio.sleep(60)  # 1분 대기
            except Exception as e:
                print(f"분석 오류: {str(e)}")
                await asyncio.sleep(60)

    async def forward_openai_messages(self):
        try:
            while True:
                if self.openai_ws:
                    message = await self.openai_ws.recv()
                    message_data = json.loads(message)

                    # response.done 타입의 메시지만 전달
                    if message_data.get("type") == "response.done":
                        await self.send(text_data=message)

        except websockets.exceptions.ConnectionClosed:
            await self.reconnect()
        except Exception as e:
            print(f"전달 오류: {str(e)}")
            await self.reconnect()

    async def reconnect(self):
        try:
            await self.initialize_openai_connection()
        except Exception as e:
            print(f"재연결 실패: {str(e)}")
            await asyncio.sleep(5)

    async def receive(self, text_data):
        try:
            if self.openai_ws:
                await self.openai_ws.send(text_data)
        except Exception as e:
            await self.send(json.dumps({
                "type": "error",
                "message": f"메시지 전송 오류: {str(e)}"
            }))

    async def disconnect(self, close_code):
        if self.analysis_task:
            self.analysis_task.cancel()
        if self.openai_ws:
            await self.openai_ws.close()