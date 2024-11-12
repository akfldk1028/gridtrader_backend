from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import StrategyConfig
from .serializers import StrategyConfigSerializer


class StrategyConfigView(APIView):
    def get(self, request, strategy_name):
        try:
            strategy = StrategyConfig.objects.get(name=strategy_name)
            serializer = StrategyConfigSerializer(strategy)
            return Response(serializer.data)
        except StrategyConfig.DoesNotExist:
            return Response({"error": "Strategy not found"}, status=status.HTTP_404_NOT_FOUND)

    def post(self, request):
        serializer = StrategyConfigSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, strategy_name):
        try:
            strategy = StrategyConfig.objects.get(name=strategy_name)
        except StrategyConfig.DoesNotExist:
            return Response({"error": "Strategy not found"}, status=status.HTTP_404_NOT_FOUND)

        serializer = StrategyConfigSerializer(strategy, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def patch(self, request, strategy_name):
        try:
            strategy = StrategyConfig.objects.get(name=strategy_name)
        except StrategyConfig.DoesNotExist:
            return Response({"error": "Strategy not found"}, status=status.HTTP_404_NOT_FOUND)

        serializer = StrategyConfigSerializer(strategy, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, strategy_name):
        try:
            strategy = StrategyConfig.objects.get(name=strategy_name)
        except StrategyConfig.DoesNotExist:
            return Response({"error": "Strategy not found"}, status=status.HTTP_404_NOT_FOUND)

        strategy.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)