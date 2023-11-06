from rest_framework.decorators import api_view
from rest_framework.request import Request 
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import permission_classes
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.tokens import AccessToken, RefreshToken
from user.models import CustomUserManager, User

import json

@api_view(['POST'])
@permission_classes([AllowAny])
def login_api(req: Request) -> Response:
    """
    The login enpoint require three argument from request is email, password, and is_doctor
    ------------------------------ 
    Input: the request from client
    ------------------------------
    Out put the response to client
    ------------------------------
    Status code:
    - 400: BAD Request -> Not enough or mis-match argumentaion
    - 200: Accepted -> Login successfully.
    """
    body: dict = req.data
    user_mangement = CustomUserManager()
    email, password, is_doctor = body.get("email"), body.get("password"), body.get("is_doctor")
    if any([e_form is None for e_form in [email, password, is_doctor]]):
        return Response(status=status.HTTP_400_BAD_REQUEST)
    is_accept, temp = user_mangement.authenticate(email=email, password=password)
    if (is_accept == 1):
        user: User = temp
        access_token, refresh_token = AccessToken.for_user(user), RefreshToken.for_user(user)
        access_token["lname"] = user.lname
        access_token["fname"] = user.fname
        access_token["is_doctor"] = is_doctor
        access_token["email"] = user.email
        return Response({
            'access_token': str(access_token),
            'refresh token': str(refresh_token), 
            },status=status.HTTP_202_ACCEPTED)
    elif (is_accept == -1):
        error = temp
        return Response(
            {'Error': error},
            status=status.HTTP_401_UNAUTHORIZED
        )
@api_view(["POST"])
def signup_api(req: Request) -> Response:
    """
    """
    body = req.data
    json_data: dict = json.load(body)
    email, password, age, name = json_data.get("email"), json_data.get("password"), json_data.get("age"), json_data.get("lname"), json_data.get("fname")
    if any([data is not None for data in [email, password, age, name]]):
        return Response(status=status.HTTP_400_BAD_REQUEST)
    # UserMangement.creaet_superuser()
    return Response(status=status.HTTP_201_CREATED)