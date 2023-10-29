from rest_framework import generics
from rest_framework.response import Response
from .serializers import FileSerializer
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework import status

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
