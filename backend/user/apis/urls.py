from django.urls import path
from .views import signup_api, login_api
from rest_framework_simplejwt.views import (
    TokenRefreshView
)
urlpatterns = [
    path("api/login/", login_api, name='login_api'),
    path("api/register/", signup_api, name='register_api'),
    path("api/token/refresh/", TokenRefreshView.as_view(), name='refresh_token'),
]