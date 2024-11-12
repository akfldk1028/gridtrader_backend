from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db import transaction
from .models import Log, Order, Strategy
from .serializers import LogSerializer, OrderSerializer, StrategySerializer
from django.core.exceptions import ValidationError
from django.db.models import Q

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.exceptions import ValidationError
from django.core.cache import cache
import json
from .serializers import LogSerializer, OrderSerializer, StrategySerializer
from django.core.cache import caches
from datetime import datetime
from datetime import timedelta
import logging

import logging
from django.core.cache import caches
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from datetime import datetime, timedelta
import json
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
import time




logger = logging.getLogger(__name__)

CACHE_TIMEOUT = 60 * 3  # 10 minutes in seconds


class BaseDataView(APIView):
    cache_name = 'logSession'
    cache_key_prefix = None

    def get_cache(self):
        try:
            cache = caches[self.cache_name]
            logger.info(f"Using cache: {self.cache_name}")
            return cache
        except Exception as e:
            logger.error(f"Failed to get cache '{self.cache_name}': {str(e)}")
            raise

    def get_cache_key(self):
        if not self.cache_key_prefix:
            raise ImproperlyConfigured("cache_key_prefix must be set in the subclass")
        today = datetime.now().strftime('%Y%m%d')
        return f"{self.cache_key_prefix}:{today}"

    def get(self, request):
        cache = self.get_cache()
        key = self.get_cache_key()

        try:
            stored_data = cache.get(key)
            if stored_data:
                decoded_data = stored_data.decode('utf-8') if isinstance(stored_data, bytes) else stored_data
                return Response(json.loads(decoded_data), status=status.HTTP_200_OK)
            else:
                return Response([], status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error retrieving data from cache for key {key}: {str(e)}")
            return Response({"error": "Failed to retrieve data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request):
        cache = self.get_cache()
        key = self.get_cache_key()
        new_data = request.data

        if not isinstance(new_data, list):
            new_data = [new_data]

        logger.info(f"Attempting to update data in cache '{self.cache_name}' with key '{key}'")
        logger.debug(f"Data to be cached: {str(new_data)[:100]}...")  # Log first 100 characters of data

        try:
            stored_data = cache.get(key)
            if stored_data:
                decoded_data = stored_data.decode('utf-8') if isinstance(stored_data, bytes) else stored_data
                existing_data = json.loads(decoded_data)
            else:
                existing_data = []

            self.merge_data(existing_data, new_data)

            encoded_data = json.dumps(existing_data, ensure_ascii=False).encode('utf-8')
            cache.set(key, encoded_data, timeout=CACHE_TIMEOUT)

            logger.info(f"Successfully updated data in cache '{self.cache_name}' with key '{key}'")
        except Exception as e:
            logger.error(f"Failed to update data in cache '{self.cache_name}' with key '{key}': {str(e)}")
            return Response({"error": "Failed to cache data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"message": "Data synchronized successfully"}, status=status.HTTP_200_OK)

    def merge_data(self, existing_data, new_data):
        existing_data.extend(new_data)

    def delete(self, request):
        cache = self.get_cache()
        key = self.get_cache_key()

        try:
            cache.delete(key)
            return Response({"message": "All items for today deleted successfully"}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Failed to delete data from cache '{self.cache_name}' with key '{key}': {str(e)}")
            return Response({"error": "Failed to delete data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class OrderDataView(BaseDataView):
    cache_key_prefix = 'order'

    def merge_data(self, existing_data, new_data):
        order_dict = {item['orderid']: item for item in existing_data if 'orderid' in item}

        for item in new_data:
            if 'orderid' in item:
                order_dict[item['orderid']] = item
            else:
                logger.warning(f"Received order data without 'orderid' key: {item}")

        existing_data.clear()
        existing_data.extend(order_dict.values())


class LogDataView(BaseDataView):
    cache_key_prefix = 'log'

    def merge_data(self, existing_data, new_data):
        existing_data.extend(new_data)


class StrategyDataView(BaseDataView):
    cache_key_prefix = 'strategy'

    def merge_data(self, existing_data, new_data):
        strategy_dict = {item['strategy_name']: item for item in existing_data if 'strategy_name' in item}

        for item in new_data:
            if 'strategy_name' in item:
                strategy_dict[item['strategy_name']] = item
            else:
                logger.warning(f"Received strategy data without 'strategy_name' key: {item}")

        existing_data.clear()
        existing_data.extend(strategy_dict.values())

# class BaseDataView(APIView):
#     cache_name = 'logSession'
#     cache_key_prefix = None
#
#     def get_cache(self):
#         try:
#             cache = caches[self.cache_name]
#             logger.info(f"Using cache: {self.cache_name}")
#             return cache
#         except Exception as e:
#             logger.error(f"Failed to get cache '{self.cache_name}': {str(e)}")
#             raise
#
#     def get_cache_key(self):
#         if not self.cache_key_prefix:
#             raise ImproperlyConfigured("cache_key_prefix must be set in the subclass")
#         today = datetime.now().strftime('%Y%m%d')
#         return f"{self.cache_key_prefix}:{today}"
#
#     def get(self, request):
#         cache = self.get_cache()
#         key = self.get_cache_key()
#
#         try:
#             stored_data = cache.get(key)
#             if stored_data:
#                 decoded_data = stored_data.decode('utf-8') if isinstance(stored_data, bytes) else stored_data
#                 return Response(json.loads(decoded_data), status=status.HTTP_200_OK)
#             else:
#                 return Response([], status=status.HTTP_200_OK)
#         except Exception as e:
#             logger.error(f"Error retrieving data from cache for key {key}: {str(e)}")
#             return Response({"error": "Failed to retrieve data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#
#     def post(self, request):
#         cache = self.get_cache()
#         key = self.get_cache_key()
#         new_data = request.data
#
#         if not isinstance(new_data, list):
#             new_data = [new_data]
#
#         logger.info(f"Attempting to update data in cache '{self.cache_name}' with key '{key}'")
#         logger.debug(f"Data to be cached: {str(new_data)[:100]}...")  # Log first 100 characters of data
#
#         try:
#             stored_data = cache.get(key)
#             if stored_data:
#                 decoded_data = stored_data.decode('utf-8') if isinstance(stored_data, bytes) else stored_data
#                 existing_data = json.loads(decoded_data)
#             else:
#                 existing_data = []
#
#             # 기존 데이터와 새 데이터 병합
#             for item in new_data:
#                 if 'orderid' in item:  # order의 경우
#                     existing_item = next((i for i in existing_data if i.get('orderid') == item['orderid']), None)
#                     if existing_item:
#                         existing_item.update(item)
#                     else:
#                         existing_data.append(item)
#                 else:  # log나 strategy의 경우
#                     existing_data.append(item)
#
#             # UTF-8로 인코딩
#             encoded_data = json.dumps(existing_data, ensure_ascii=False).encode('utf-8')
#             cache.set(key, encoded_data, timeout=CACHE_TIMEOUT)
#
#             logger.info(f"Successfully updated data in cache '{self.cache_name}' with key '{key}'")
#         except Exception as e:
#             logger.error(f"Failed to update data in cache '{self.cache_name}' with key '{key}': {str(e)}")
#             return Response({"error": "Failed to cache data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#
#         return Response({"message": "Data synchronized successfully"}, status=status.HTTP_200_OK)
#
#     def delete(self, request):
#         cache = self.get_cache()
#         key = self.get_cache_key()
#
#         try:
#             cache.delete(key)
#             return Response({"message": "All items for today deleted successfully"}, status=status.HTTP_200_OK)
#         except Exception as e:
#             logger.error(f"Failed to delete data from cache '{self.cache_name}' with key '{key}': {str(e)}")
#             return Response({"error": "Failed to delete data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#
#
# class OrderDataView(BaseDataView):
#     cache_key_prefix = 'order'
#
#
# class LogDataView(BaseDataView):
#     cache_key_prefix = 'log'
#
#
# class StrategyDataView(BaseDataView):
#     cache_key_prefix = 'strategy'
#
