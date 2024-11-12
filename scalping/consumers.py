# consumers.py
from channels.generic.websocket import AsyncWebsocketConsumer
import json
import websockets
import asyncio
from django.conf import settings


class OpenAIConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.openai_ws = None
        await self.accept()
        await self.initialize_openai_connection()

    async def initialize_openai_connection(self):
        try:
            # OpenAI WebSocket URL
            url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

            # 연결 설정
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }

            self.openai_ws = await websockets.connect(url, extra_headers=headers)

            # 초기 메시지 전송
            await self.openai_ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "instructions": "Please assist the user."
                }
            }))

            # OpenAI로부터의 메시지를 클라이언트에게 전달하는 태스크 시작
            asyncio.create_task(self.forward_openai_messages())

        except Exception as e:
            await self.send(json.dumps({
                "type": "error",
                "message": f"Failed to connect to OpenAI: {str(e)}"
            }))
            await self.close()

    async def forward_openai_messages(self):
        try:
            while True:
                if self.openai_ws:
                    message = await self.openai_ws.recv()
                    await self.send(text_data=message)
        except websockets.exceptions.ConnectionClosed:
            await self.send(json.dumps({
                "type": "error",
                "message": "OpenAI connection closed"
            }))
        except Exception as e:
            await self.send(json.dumps({
                "type": "error",
                "message": f"Error forwarding message: {str(e)}"
            }))

    async def receive(self, text_data):
        try:
            if self.openai_ws:
                await self.openai_ws.send(text_data)
        except Exception as e:
            await self.send(json.dumps({
                "type": "error",
                "message": f"Error sending message: {str(e)}"
            }))

    async def disconnect(self, close_code):
        if self.openai_ws:
            await self.openai_ws.close()