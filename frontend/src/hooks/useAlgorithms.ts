import { useQuery } from '@tanstack/react-query';
import { apiService } from '@/services/api';
import { AlgorithmCategory } from '@/types';

export function useAlgorithms(category?: AlgorithmCategory) {
  return useQuery({
    queryKey: ['algorithms', category],
    queryFn: () => apiService.getAlgorithms(category || 'ml'),
    enabled: !!category,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

export function useAlgorithmById(algorithmId: string) {
  return useQuery({
    queryKey: ['algorithm', algorithmId],
    queryFn: () => apiService.getAlgorithmById(algorithmId),
    enabled: !!algorithmId,
  });
}

export function useCategories() {
  return useQuery({
    queryKey: ['categories'],
    queryFn: () => apiService.getCategories(),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}
