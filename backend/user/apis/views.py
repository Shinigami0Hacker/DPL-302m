from rest_framework.decorators import api_view
from rest_framework.request import Request 
from rest_framework.response import Response
from rest_framework import status

import json

@api_view(['GET'])
def login_api(req: Request):
    """
    """
    body = req.data
    json_data: dict = json.load(body)
    email, password = json_data.get("email"), json_data.get("password")
    if email or password:
        return Response(status=status.HTTP_400_BAD_REQUEST)
    
    return Response({
        'Authetication': "jwt_token"
        },status=status.HTTP_200_OK)

@api_view(["GET"])
def signup_api(req: Request):
    """
    """
    body = req.data
    json_data: dict = json.load(body)
    email, password, age, name = json_data.get("email"), json_data.get("password"), json_data.get("age"), json_data.get("email")
    if any([data is not None for data in [email, password, age, name]]):
        return Response(status=status.HTTP_400_BAD_REQUEST)
    # UserMangement.creaet_superuser()
    return Response(status=status.HTTP_201_CREATED)