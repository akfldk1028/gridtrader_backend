import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .models import Log, Order, Strategy
from asgiref.sync import sync_to_async

class DataConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Simple token-based authentication
        token = self.scope['url_route']['kwargs'].get('token')
        if token != 'your_secret_token':  # Replace with a secure token
            await self.close()
        else:
            await self.accept()

    async def disconnect(self, close_code):
        pass

    @sync_to_async
    def get_latest_data(self):
        latest_logs = list(Log.objects.order_by('-time')[:10].values())
        latest_orders = list(Order.objects.order_by('-datetime')[:10].values())
        latest_strategies = list(Strategy.objects.order_by('-id')[:10].values())
        return {
            'logs': latest_logs,
            'orders': latest_orders,
            'strategies': latest_strategies
        }

    async def receive(self, text_data):
        data = await self.get_latest_data()
        await self.send(text_data=json.dumps(data))



# import React, { useState, useEffect } from 'react';
#
# function App() {
#   const [data, setData] = useState({ logs: [], orders: [], strategies: [] });
#
#   useEffect(() => {
#     const token = 'your_secret_token';  // Replace with your actual token
#     const socket = new WebSocket(`ws://localhost:8000/ws/data/${token}/`);
#
#     socket.onopen = () => {
#       console.log('WebSocket connection established');
#       socket.send(JSON.stringify({ action: 'get_data' }));
#     };
#
#     socket.onmessage = (event) => {
#       const newData = JSON.parse(event.data);
#       setData(newData);
#     };
#
#     socket.onclose = () => {
#       console.log('WebSocket connection closed');
#     };
#
#     const interval = setInterval(() => {
#       if (socket.readyState === WebSocket.OPEN) {
#         socket.send(JSON.stringify({ action: 'get_data' }));
#       }
#     }, 5000);
#
#     return () => {
#       clearInterval(interval);
#       socket.close();
#     };
#   }, []);
#
#   // Render your data here
#   return (
#     <div>
#       {/* Render logs, orders, and strategies data */}
#     </div>
#   );
# }
#
# export default App;