import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { AlgorithmLayout } from '@/components/common/AlgorithmLayout';
import Button from '@/components/common/Button';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { ErrorDisplay } from '@/components/common/ErrorDisplay';
import { apiService } from '@/services/api';
import { Controls } from './Controls';
import { Visualization } from './Visualization';
import { Documentation } from './Documentation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface HierarchicalClusteringParams {
  n_clusters: number;
  linkage: string;
  affinity: string;
  n_samples: number;
}

interface TrainingResult {
  metrics: {
    silhouette_score: number;
    davies_bouldin_score: number;
    calinski_harabasz_score: number;
  };
  predictions: number[];
  visualization_data: {
    scatter: {
      X: number[][];
      labels: number[];
      n_clusters: number;
    };
    dendrogram: {
      linkage_matrix: number[][];
      n_clusters: number;
      linkage_method: string;
      affinity: string;
    };
  };
  execution_time_ms: number;
  model_info: {
    n_clusters: number;
    linkage: string;
    affinity: string;
    n_samples: number;
    cluster_sizes: Record<string, number>;
  };
}

export function HierarchicalClusteringDemo() {
  const [parameters, setParameters] = useState<HierarchicalClusteringParams>({
    n_clusters: 3,
    linkage: 'ward',
    affinity: 'euclidean',
    n_samples: 300,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['hierarchical-clustering-info'],
    queryFn: () => apiService.getAlgorithmInfo('ml', 'hierarchical-clustering'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: HierarchicalClusteringParams) =>
      apiService.trainAlgorithm('ml', 'hierarchical-clustering', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof HierarchicalClusteringParams, value: any) => {
    setParameters((prev) => ({
      ...prev,
      [name]: value,
    }));

    // Reset affinity to euclidean when switching to ward linkage
    if (name === 'linkage' && value === 'ward') {
      setParameters((prev) => ({
        ...prev,
        linkage: value,
        affinity: 'euclidean',
      }));
    }
  };

  if (isLoadingInfo) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <AlgorithmLayout
      title="Hierarchical Clustering"
      description="Clustering method that builds a hierarchy of clusters using linkage"
      category="Machine Learning"
      algorithmInfo={algorithmInfo}
    >
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Panel */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Parameters</CardTitle>
            </CardHeader>
            <CardContent>
              <Controls
                parameters={parameters}
                onChange={handleParameterChange as (name: string, value: any) => void}
                disabled={trainMutation.isPending}
              />
              <Button
                onClick={handleTrain}
                disabled={trainMutation.isPending}
                className="w-full mt-4"
                size="lg"
              >
                {trainMutation.isPending ? 'Training...' : 'Train Model'}
              </Button>
            </CardContent>
          </Card>

          {/* Metrics Display */}
          {result && (
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Clustering Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Silhouette Score</span>
                  <span className="font-mono font-semibold">
                    {result.metrics.silhouette_score.toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Davies-Bouldin Index</span>
                  <span className="font-mono font-semibold">
                    {result.metrics.davies_bouldin_score.toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Calinski-Harabasz Score</span>
                  <span className="font-mono font-semibold">
                    {result.metrics.calinski_harabasz_score.toFixed(2)}
                  </span>
                </div>
                <div className="pt-3 border-t">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Execution Time</span>
                    <span className="font-mono font-semibold">
                      {result.execution_time_ms.toFixed(2)}ms
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Visualization Panel */}
        <div className="lg:col-span-2">
          {trainMutation.isPending && (
            <div className="flex items-center justify-center h-96">
              <LoadingSpinner size="lg" />
            </div>
          )}

          {trainMutation.isError && (
            <ErrorDisplay
              message={
                trainMutation.error instanceof Error
                  ? trainMutation.error.message
                  : 'Training failed'
              }
            />
          )}

          {result && !trainMutation.isPending && (
            <Visualization result={result} />
          )}

          {!result && !trainMutation.isPending && !trainMutation.isError && (
            <Card>
              <CardContent className="flex items-center justify-center h-96">
                <p className="text-muted-foreground">
                  Configure parameters and click "Train Model" to see results
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Documentation */}
      <div className="mt-8">
        <Documentation />
      </div>
    </AlgorithmLayout>
  );
}

export default HierarchicalClusteringDemo;
