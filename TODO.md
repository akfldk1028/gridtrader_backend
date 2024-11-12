## django-q
각 task 응답도 postgre로
매일 한번씩 수익률 계산



## redis 객체들 전부 get 만들고


## postgre get
## react 무한루프 


import { useQuery, useMutation, useQueryClient } from 'react-query';
import axios from 'axios';

// 타입 정의
type Order = {
  orderid: string;
  // 다른 필드들 추가
};

type Log = {
  // Log 관련 필드들 정의
};

type Strategy = {
  strategy_name: string;
  // 다른 필드들 추가
};

// API 함수들
const fetchData = async <T>(endpoint: string): Promise<T[]> => {
  const response = await axios.get<T[]>(`/api/${endpoint}`);
  return response.data;
};

const postData = async <T>(endpoint: string, data: T[]): Promise<void> => {
  await axios.post(`/api/${endpoint}`, data);
};

const deleteData = async (endpoint: string): Promise<void> => {
  await axios.delete(`/api/${endpoint}`);
};

// Custom Hooks
export const useOrderData = () => {
  const queryClient = useQueryClient();

  const query = useQuery<Order[], Error>('orders', () => fetchData<Order>('order'));

  const mutation = useMutation<void, Error, Order[]>(
    (newOrders) => postData<Order>('order', newOrders),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('orders');
      },
    }
  );

  const deleteMutation = useMutation<void, Error>(
    () => deleteData('order'),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('orders');
      },
    }
  );

  return {
    orders: query.data ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    updateOrders: mutation.mutate,
    deleteOrders: deleteMutation.mutate,
  };
};

export const useLogData = () => {
  const queryClient = useQueryClient();

  const query = useQuery<Log[], Error>('logs', () => fetchData<Log>('log'));

  const mutation = useMutation<void, Error, Log[]>(
    (newLogs) => postData<Log>('log', newLogs),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('logs');
      },
    }
  );

  const deleteMutation = useMutation<void, Error>(
    () => deleteData('log'),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('logs');
      },
    }
  );

  return {
    logs: query.data ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    updateLogs: mutation.mutate,
    deleteLogs: deleteMutation.mutate,
  };
};

export const useStrategyData = () => {
  const queryClient = useQueryClient();

  const query = useQuery<Strategy[], Error>('strategies', () => fetchData<Strategy>('strategy'));

  const mutation = useMutation<void, Error, Strategy[]>(
    (newStrategies) => postData<Strategy>('strategy', newStrategies),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('strategies');
      },
    }
  );

  const deleteMutation = useMutation<void, Error>(
    () => deleteData('strategy'),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('strategies');
      },
    }
  );

  return {
    strategies: query.data ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    updateStrategies: mutation.mutate,
    deleteStrategies: deleteMutation.mutate,
  };
};