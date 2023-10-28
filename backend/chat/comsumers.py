import json
from channels.generic.websocket import WebsocketConsumer, AsyncWebsocketConsumer
from asgiref.sync import async_to_sync
import asyncio
import cv2

class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.room_group_name = 'test'

        async_to_sync(self.channel_layer.group_add)(
            self.room_group_name,
            self.channel_name
        )
        self.accept()

    def chat_message(self, event):
        message = event['message']

        self.send(text_data=json.dumps({
            'type':'chat',
            'message':message
        }))

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        async_to_sync(self.channel_layer.group_send)(
            self.room_group_name,
            {
                'type':'chat_message',
                'message':message
            }
        )


async def video_producer():
    channel_layer = get_channel_layer()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        await async_to_sync(channel_layer.group_send)("video_stream", {"type": "video.stream", "frame": frame_bytes})
        await asyncio.sleep(0.1)

class VideoComsumer(AsyncWebsocketConsumer):
    async def connect(self):
        return await super().connect()
    
    async def disconnect(self, code):
        return await super().disconnect(code)
    
    async def video_stream(self, event):
        return await self.send(event["frame"])


    

