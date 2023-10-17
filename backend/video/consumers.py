# from channels.generic.websocket import AsyncWebsocketConsumer
# import cv2
# import base64
# import numpy as np
# import json

# class VideoConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         self.accept()

#         self.send(text_data=json.dumps({
#             'type': 'connection_established',
#             'message': 'You are connected'
#         }))
#         return super().connect()
    
#     async def disconnect(self, code):
#         return await super().disconnect(code)
    

#     async def receive(self, text_data=None, bytes_data=None):
#         frame_bytes = base64.b64decode(text_data)

#         frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

#         await self.send("Done")
#         return await super().receive(text_data, bytes_data)