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

interface KNNParams {
  n_neighbors: number;
  weights: 'uniform' | 'distance';
  metric: 'euclidean' | 'manhattan' | 'minkowski';
  p: number;
  test_size: number;
  random_state: number;
}

interface KNNMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  train_accuracy: number;
  test_accuracy: number;
}

interface VisualizationData {
  scatter_data: Array<{
    x: number;
    y: number;
    class: number;
    predicted: number;
    label: string;
  }>;
  confusion_matrix: number[][];
  class_labels: string[];
  decision_boundary?: Array<{
    x: number;
    y: number;
    class: number;
  }>;
}

interface TrainingResult {
  success: boolean;
  metrics: KNNMetrics;
  predictions: number[];
  actual: number[];
  visualization_data: VisualizationData;
  execution_time_ms: number;
  parameters: Record<string, any>;
  message?: string;
  error?: string;
}

export function KNNDemo() {
  const [parameters, setParameters] = useState<KNNParams>({
    n_neighbors: 5,
    weights: 'uniform',
    metric: 'euclidean',
    p: 2,
    test_size: 0.3,
    random_state: 42,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['knn-info'],
    queryFn: () => apiService.getAlgorithmInfo('ml', 'knn'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: KNNParams) =>
      apiService.trainAlgorithm('ml', 'knn', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof KNNParams, value: any) => {
    setParameters((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  if (isLoadingInfo) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <AlgorithmLayout
      title="K-Nearest Neighbors"
      description={algorithmInfo?.metadata?.description}
      category="Machine Learning"
      sections={{
        parameters: (
          <div className="space-y-4">
            <Controls
              parameters={parameters}
              onChange={handleParameterChange as (name: string, value: any) => void}
              disabled={trainMutation.isPending}
            />
            <Button
              onClick={handleTrain}
              disabled={trainMutation.isPending}
              className="w-full"
            >
              {trainMutation.isPending ? 'Training...' : 'Train Model'}
            </Button>
          </div>
        ),
        visualization: (
          <div className="space-y-4">
            {trainMutation.isError && (
              <ErrorDisplay
                error={
                  trainMutation.error instanceof Error
                    ? trainMutation.error.message
                    : 'Training failed'
                }
              />
            )}
            {trainMutation.isPending && (
              <div className="flex items-center justify-center h-96">
                <LoadingSpinner />
              </div>
            )}
            {result && !trainMutation.isPending && (
              <Visualization result={result} />
            )}
            {!result && !trainMutation.isPending && !trainMutation.isError && (
              <div className="flex items-center justify-center h-96 text-muted-foreground">
                Train the model to see visualization
              </div>
            )}
          </div>
        ),
        results: result && (
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Accuracy</p>
                    <p className="text-2xl font-bold">
                      {(result.metrics.accuracy * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Precision</p>
                    <p className="text-2xl font-bold">
                      {(result.metrics.precision * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Recall</p>
                    <p className="text-2xl font-bold">
                      {(result.metrics.recall * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">F1 Score</p>
                    <p className="text-2xl font-bold">
                      {(result.metrics.f1_score * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Train Accuracy</p>
                      <p className="font-semibold">
                        {(result.metrics.train_accuracy * 100).toFixed(2)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Test Accuracy</p>
                      <p className="font-semibold">
                        {(result.metrics.test_accuracy * 100).toFixed(2)}%
                      </p>
                    </div>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t">
                  <p className="text-sm text-muted-foreground">
                    Execution Time: {result.execution_time_ms.toFixed(2)}ms
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Model Parameters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="font-medium">Number of Neighbors (K):</span>
                    <span>{result.parameters.n_neighbors}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Weight Function:</span>
                    <span className="capitalize">{result.parameters.weights}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Distance Metric:</span>
                    <span className="capitalize">{result.parameters.metric}</span>
                  </div>
                  {result.parameters.metric === 'minkowski' && (
                    <div className="flex justify-between">
                      <span className="font-medium">Minkowski Power (p):</span>
                      <span>{result.parameters.p}</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="font-medium">Test Size:</span>
                    <span>{(result.parameters.test_size * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        ),
      }}
    >
      <Documentation metadata={algorithmInfo?.metadata} />
    </AlgorithmLayout>
  );
}

export default KNNDemo;
