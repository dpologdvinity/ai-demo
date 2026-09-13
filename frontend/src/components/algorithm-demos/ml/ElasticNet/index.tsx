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

interface ElasticNetParams {
  alpha: number;
  l1_ratio: number;
  max_iter: number;
  fit_intercept: boolean;
  normalize: boolean;
  dataset_name: string;
  test_size: number;
}

interface TrainingResult {
  success: boolean;
  metrics: {
    r2_score: number;
    mse: number;
    rmse: number;
    mae: number;
  };
  visualization_data: {
    chart_data: Array<{
      index: number;
      predicted: number;
      actual: number;
    }>;
    coefficient_data: Array<{
      feature: string;
      coefficient: number;
      magnitude: number;
      selected: boolean;
    }>;
    regularization_path: Array<{
      l1_ratio: number;
      type: string;
      r2_score: number;
      n_nonzero: number;
      sparsity: number;
      l2_norm: number;
    }>;
  };
  execution_time_ms: number;
  model_info: {
    alpha: number;
    l1_ratio: number;
    coefficients: number[];
    intercept: number;
    n_features_in: number;
    n_nonzero_coefs: number;
    sparsity: number;
    l2_norm: number;
    converged: boolean;
  };
  feature_importance: Record<string, number>;
  parameters_used: Record<string, any>;
  error?: string;
}

export function ElasticNetDemo() {
  const [parameters, setParameters] = useState<ElasticNetParams>({
    alpha: 1.0,
    l1_ratio: 0.5,
    max_iter: 1000,
    fit_intercept: true,
    normalize: true,
    dataset_name: 'boston',
    test_size: 0.3,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['elastic-net-info'],
    queryFn: () => apiService.getAlgorithmInfo('ml', 'elastic-net'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: ElasticNetParams) =>
      apiService.trainAlgorithm('ml', 'elastic-net', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof ElasticNetParams, value: any) => {
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
      title="Elastic Net"
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
                    <p className="text-sm text-muted-foreground">R² Score</p>
                    <p className="text-2xl font-bold">
                      {result.metrics.r2_score.toFixed(4)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">MSE</p>
                    <p className="text-2xl font-bold">
                      {result.metrics.mse.toFixed(4)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">RMSE</p>
                    <p className="text-2xl font-bold">
                      {result.metrics.rmse.toFixed(4)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">MAE</p>
                    <p className="text-2xl font-bold">
                      {result.metrics.mae.toFixed(4)}
                    </p>
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
                <CardTitle>Regularization Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="grid grid-cols-2 gap-4 pb-4 border-b">
                    <div>
                      <span className="font-medium">Selected Features:</span>{' '}
                      {result.model_info.n_nonzero_coefs} / {result.model_info.n_features_in}
                    </div>
                    <div>
                      <span className="font-medium">Sparsity:</span>{' '}
                      {result.model_info.sparsity.toFixed(1)}%
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="font-medium">L1 Ratio:</span>{' '}
                      {result.model_info.l1_ratio.toFixed(2)}
                    </div>
                    <div>
                      <span className="font-medium">Alpha (α):</span>{' '}
                      {result.model_info.alpha.toFixed(3)}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="font-medium">L2 Norm:</span>{' '}
                      {result.model_info.l2_norm.toFixed(4)}
                    </div>
                    <div>
                      <span className="font-medium">Converged:</span>{' '}
                      {result.model_info.converged ? 'Yes' : 'No'}
                    </div>
                  </div>
                  <div>
                    <span className="font-medium">Intercept:</span>{' '}
                    {result.model_info.intercept.toFixed(4)}
                  </div>
                  <div className="pt-2">
                    <span className="font-medium">Non-zero Coefficients:</span>
                    <div className="mt-2 max-h-40 overflow-y-auto space-y-1">
                      {result.visualization_data.coefficient_data
                        .filter((item) => item.selected)
                        .map((item, idx) => (
                          <div key={idx} className="text-xs text-muted-foreground flex justify-between">
                            <span>{item.feature}</span>
                            <span className="font-mono">{item.coefficient.toFixed(6)}</span>
                          </div>
                        ))}
                    </div>
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

export default ElasticNetDemo;
