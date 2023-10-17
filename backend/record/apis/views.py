#External Importing
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.decorators import permission_classes, api_view
from rest_framework.permissions import IsAuthenticated 

#Internal Importing

@permission_classes([IsAuthenticated])
@api_view(['GET'])
def get_model_result_records():
    """
    
    
    """
    pass



@permission_classes([IsAuthenticated])
@api_view(['GET'])
def get_model_result(req: Request, record_id: id):
    """
    
    """
    pass