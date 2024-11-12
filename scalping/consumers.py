# # consumers.py
# from channels.generic.websocket import AsyncWebsocketConsumer
# import json
# import websockets
# import asyncio
# from django.conf import settings
#
#
# class OpenAIConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         self.openai_ws = None
#         await self.accept()
#         await self.initialize_openai_connection()
#
#     async def initialize_openai_connection(self):
#         try:
#             # OpenAI WebSocket URL
#             url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
#
#             # 연결 설정
#             headers = {
#                 "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
#                 "OpenAI-Beta": "realtime=v1"
#             }
#
#             self.openai_ws = await websockets.connect(url, extra_headers=headers)
#
#             # 초기 메시지 전송
#             await self.openai_ws.send(json.dumps({
#                 "type": "response.create",
#                 "response": {
#                     "modalities": ["text"],
#                     "instructions": "Please assist the user."
#                 }
#             }))
#
#             # OpenAI로부터의 메시지를 클라이언트에게 전달하는 태스크 시작
#             asyncio.create_task(self.forward_openai_messages())
#
#         except Exception as e:
#             await self.send(json.dumps({
#                 "type": "error",
#                 "message": f"Failed to connect to OpenAI: {str(e)}"
#             }))
#             await self.close()
#
#     async def forward_openai_messages(self):
#         try:
#             while True:
#                 if self.openai_ws:
#                     message = await self.openai_ws.recv()
#                     await self.send(text_data=message)
#         except websockets.exceptions.ConnectionClosed:
#             await self.send(json.dumps({
#                 "type": "error",
#                 "message": "OpenAI connection closed"
#             }))
#         except Exception as e:
#             await self.send(json.dumps({
#                 "type": "error",
#                 "message": f"Error forwarding message: {str(e)}"
#             }))
#
#     async def receive(self, text_data):
#         try:
#             if self.openai_ws:
#                 await self.openai_ws.send(text_data)
#         except Exception as e:
#             await self.send(json.dumps({
#                 "type": "error",
#                 "message": f"Error sending message: {str(e)}"
#             }))
#
#     async def disconnect(self, close_code):
#         if self.openai_ws:
#             await self.openai_ws.close()

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
        # Start the periodic analysis task
        self.analysis_task = asyncio.create_task(self.periodic_bitcoin_analysis())

    async def initialize_openai_connection(self):
        try:
            url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }

            self.openai_ws = await websockets.connect(url, extra_headers=headers)

            # Initialize with analysis-focused instructions
            await self.openai_ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "instructions": """
                    Analyze Bitcoin price movements and market conditions.
                    Provide concise insights focusing on:
                    - Price changes
                    - Volume analysis
                    - Key support/resistance levels
                    - Notable market events
                    Keep responses brief and actionable.
                    """
                }
            }))

            asyncio.create_task(self.forward_openai_messages())

        except Exception as e:
            await self.send(json.dumps({
                "type": "error",
                "message": f"Failed to connect to OpenAI: {str(e)}"
            }))
            await self.close()

    async def get_bitcoin_data(self):
        """Fetch current Bitcoin market data"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true') as response:
                    data = await response.json()
                    return data['bitcoin']
        except Exception as e:
            return None

    async def periodic_bitcoin_analysis(self):
        """Perform periodic Bitcoin analysis"""
        while True:
            try:
                if self.openai_ws and self.openai_ws.open:
                    bitcoin_data = await self.get_bitcoin_data()

                    if bitcoin_data:
                        analysis_request = {
                            "type": "message",
                            "content": f"""
                            Current Bitcoin Data:
                            Price: ${bitcoin_data.get('usd', 'N/A')}
                            24h Change: {bitcoin_data.get('usd_24h_change', 'N/A')}%
                            24h Volume: ${bitcoin_data.get('usd_24h_vol', 'N/A')}

                            Please provide a brief market analysis based on this data.
                            """
                        }

                        await self.openai_ws.send(json.dumps(analysis_request))

                await asyncio.sleep(60)  # Wait for 1 minute
            except Exception as e:
                print(f"Analysis error: {str(e)}")
                await asyncio.sleep(60)  # Still wait before retrying

    async def forward_openai_messages(self):
        try:
            while True:
                if self.openai_ws:
                    message = await self.openai_ws.recv()
                    message_data = json.loads(message)

                    # Only forward complete responses
                    if message_data.get("type") == "response.done":
                        await self.send(text_data=message)

        except websockets.exceptions.ConnectionClosed:
            await self.reconnect()
        except Exception as e:
            print(f"Forward error: {str(e)}")
            await self.reconnect()

    async def reconnect(self):
        """Handle reconnection logic"""
        try:
            await self.initialize_openai_connection()
        except Exception as e:
            print(f"Reconnection failed: {str(e)}")
            await asyncio.sleep(5)  # Wait before retrying

    async def receive(self, text_data):
        """Handle incoming messages from client"""
        try:
            if self.openai_ws:
                await self.openai_ws.send(text_data)
        except Exception as e:
            await self.send(json.dumps({
                "type": "error",
                "message": f"Error sending message: {str(e)}"
            }))

    async def disconnect(self, close_code):
        """Clean up on disconnect"""
        if self.analysis_task:
            self.analysis_task.cancel()
        if self.openai_ws:
            await self.openai_ws.close()