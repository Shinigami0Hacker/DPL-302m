from django.urls import path
from .views import ExampleViewSet

urlpatterns = [
    path("download/<str:file_history>/", ExampleViewSet.as_view(), name="download")
]
