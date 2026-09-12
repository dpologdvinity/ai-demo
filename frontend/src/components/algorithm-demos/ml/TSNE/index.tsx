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

interface TSNEParams {
  n_components: number;
  perplexity: number;
  learning_rate: number;
  n_iter: number;
  random_state: number;
}

interface TrainingResult {
  success: boolean;
  embedded_data: number[][];
  labels: number[];
  kl_divergence: number;
  visualization_data: {
    type: '2d' | '3d';
    scatter_data: Array<{
      x: number;
      y: number;
      z?: number;
      label: number;
    }>;
    x_label: string;
    y_label: string;
    z_label?: string;
  };
  execution_time_ms: number;
  model_info: {
    n_components: number;
    perplexity: number;
    learning_rate: number;
    n_iter: number;
    n_iter_final: number;
    kl_divergence: number;
  };
  parameters_used: Record<string, any>;
  error?: string;
}

export function TSNEDemo() {
  const [parameters, setParameters] = useState<TSNEParams>({
    n_components: 2,
    perplexity: 30.0,
    learning_rate: 200.0,
    n_iter: 1000,
    random_state: 42,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['tsne-info'],
    queryFn: () => apiService.getAlgorithmInfo('ml', 'tsne'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: TSNEParams) =>
      apiService.trainAlgorithm('ml', 'tsne', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof TSNEParams, value: any) => {
    setParameters((prev) => ({
      ...prev,
      [name]: value,
    }));
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
      title="t-SNE (t-Distributed Stochastic Neighbor Embedding)"
      description="Non-linear dimensionality reduction for visualization of high-dimensional data"
      category="Machine Learning"
      difficulty="Intermediate"
    >
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Panel */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Controls
                parameters={parameters}
                onChange={handleParameterChange}
                algorithmInfo={algorithmInfo}
              />
              <Button
                onClick={handleTrain}
                disabled={trainMutation.isPending}
                variant="primary"
                className="w-full"
              >
                {trainMutation.isPending ? (
                  <>
                    <LoadingSpinner size="sm" className="mr-2" />
                    Running t-SNE...
                  </>
                ) : (
                  'Run t-SNE'
                )}
              </Button>

              {trainMutation.error && (
                <ErrorDisplay
                  error={trainMutation.error}
                  title="Training Error"
                />
              )}

              {result && result.success && (
                <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                    Results
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">KL Divergence:</span>{' '}
                      {result.kl_divergence.toFixed(4)}
                    </p>
                    <p>
                      <span className="font-medium">Execution Time:</span>{' '}
                      {result.execution_time_ms.toFixed(2)} ms
                    </p>
                    <p>
                      <span className="font-medium">Iterations:</span>{' '}
                      {result.model_info.n_iter_final}
                    </p>
                    <p>
                      <span className="font-medium">Data Points:</span>{' '}
                      {result.labels.length}
                    </p>
                    <p>
                      <span className="font-medium">Dimensions:</span>{' '}
                      {result.model_info.n_components}D
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="mt-6">
            <Documentation algorithmInfo={algorithmInfo} />
          </div>
        </div>

        {/* Visualization Panel */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Visualization</CardTitle>
            </CardHeader>
            <CardContent>
              {trainMutation.isPending ? (
                <div className="flex items-center justify-center h-96">
                  <LoadingSpinner size="lg" />
                </div>
              ) : result && result.success ? (
                <Visualization result={result} />
              ) : (
                <div className="flex items-center justify-center h-96 text-gray-500 dark:text-gray-400">
                  <div className="text-center">
                    <p className="text-lg font-medium mb-2">No Results Yet</p>
                    <p className="text-sm">
                      Configure parameters and click "Run t-SNE" to visualize high-dimensional data
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </AlgorithmLayout>
  );
}

export default TSNEDemo;
