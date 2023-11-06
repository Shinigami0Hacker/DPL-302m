from rest_framework import generics
from rest_framework.response import Response
from .serializers import FileSerializer
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from user.apis.serializers import UserSerializer

@permission_classes([IsAuthenticated])
class DownloadZipView(generics.GenericAPIView):
    """
    
    """
    serializer_class = FileSerializer
    def get(self, req: Request, *args, **kwargs) -> Response:
        file_location = self.kwargs.get("file_history")
        if file_location:
            file_respone = None
            return Response(file_respone, status=status.HTTP_200_OK, content_type="application/zip")
        return Response(status=status.HTTP_400_BAD_REQUEST)

@permission_classes([IsAuthenticated])
class PatientList(generics.ListAPIView):

    

    
