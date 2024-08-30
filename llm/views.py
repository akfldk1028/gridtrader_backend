from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import AnalysisResult
from .serializers import AnalysisResultSerializer

class LatestAnalysisResultView(APIView):
    def get(self, request):
        try:
            latest_result = AnalysisResult.objects.latest('date')
            serializer = AnalysisResultSerializer(latest_result)
            return Response(serializer.data)
        except AnalysisResult.DoesNotExist:
            return Response({"error": "No analysis results found"}, status=status.HTTP_404_NOT_FOUND)

class LatestSymbolResultView(APIView):

    def get(self, request):
        symbol = request.query_params.get('symbol')
        if not symbol:
            return Response({"error": "Symbol parameter is required."}, status=status.HTTP_400_BAD_REQUEST)

        latest_result = AnalysisResult.objects.filter(symbol=symbol).order_by('-date').first()
        if not latest_result:
            return Response({"error": "No analysis result found for the given symbol."}, status=status.HTTP_404_NOT_FOUND)

        serializer = AnalysisResultSerializer(latest_result)
        return Response(serializer.data)