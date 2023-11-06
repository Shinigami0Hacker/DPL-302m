from django.urls import re_path 
from . import comsumers

websocket_urlpatterns = [
    re_path(r'ws/chat_stream/', comsumers.ChatConsumer.as_asgi()),
    re_path(r'ws/video_stream/$', comsumers.VideoComsumer.as_asgi()),
]