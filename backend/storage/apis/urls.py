from django.urls import path
from .views import ExampleViewSet

urlpatterns = [
    path("download/<str:history_id>/", ExampleViewSet.as_view(), name="download"),
    # path("get/patient/", ,name="patient")
]
