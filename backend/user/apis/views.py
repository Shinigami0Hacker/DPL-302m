#External importing
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.decorators import api_view, permission_classes
from rest_framework import status
from rest_framework.permissions import IsAuthenticated

#Internal importing
from user.models import Doctor, Patient
from user.apis.serializers import PatientSerializer, DoctorSerializer


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_patient_list(req: Request) -> Response:
    """
    Get the list of patient responsible by the specify the doctor.
    @decorator:
    api_views: the template if access the api by browser.

    @params:
    Request: The HTTP request

    @returns:
    """

    obj_docter = Doctor.objects.get(name = "")
    patients_responsible = Patient.objects.filter(responsible_by=obj_docter)
    serializer = PatientSerializer(patients_responsible)

    return Response(
        serializer.data
        ,status=status.HTTP_200_OK
    )

