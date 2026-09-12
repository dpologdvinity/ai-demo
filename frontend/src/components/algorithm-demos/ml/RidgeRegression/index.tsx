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

interface RidgeRegressionParams {
  alpha: number;
  fit_intercept: boolean;
  solver: string;
}

interface TrainingResult {
  metrics: {
    r2_score: number;
    mse: number;
    rmse: number;
    mae: number;
  };
  predictions: number[];
  actual_values: number[];
  coefficients: Record<string, number>;
  coefficient_analysis: {
    coefficients: Record<string, number>;
    coefficients_abs: Record<string, number>;
    max_coefficient: {
      feature: string;
      value: number;
    };
    min_coefficient: {
      feature: string;
      value: number;
    };
    mean_coefficient_abs: number;
  };
  visualization_data: {
    predictions_vs_actual: Array<{
      index: number;
      predicted: number;
      actual: number;
    }>;
    coefficient_plot: Array<{
      feature: string;
      coefficient: number;
      coefficient_abs: number;
    }>;
    x_label: string;
    y_label: string;
    title: string;
  };
  execution_time_ms: number;
  model_info: {
    fit_intercept: boolean;
    alpha: number;
    solver: string;
    coefficients: number[];
    intercept: number;
    n_features_in: number;
    coef_l2_norm: number;
  };
  error?: string;
}

export function RidgeRegressionDemo() {
  const [parameters, setParameters] = useState<RidgeRegressionParams>({
    alpha: 1.0,
    fit_intercept: true,
    solver: 'auto',
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['ridge-regression-info'],
    queryFn: () => apiService.getAlgorithmInfo('ml', 'ridge-regression'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: RidgeRegressionParams) =>
      apiService.trainAlgorithm('ml', 'ridge-regression', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof RidgeRegressionParams, value: any) => {
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
      title="Ridge Regression"
      description={algorithmInfo?.metadata?.description}
      sections={{
        parameters: (
          <div className="space-y-4">
            <Controls
              parameters={parameters}
              onChange={handleParameterChange}
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
                <CardTitle>Model Information</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="font-medium">Alpha (Regularization):</span>{' '}
                    {result.model_info.alpha.toFixed(4)}
                  </div>
                  <div>
                    <span className="font-medium">Solver:</span>{' '}
                    {result.model_info.solver}
                  </div>
                  <div>
                    <span className="font-medium">Intercept:</span>{' '}
                    {result.model_info.intercept.toFixed(4)}
                  </div>
                  <div>
                    <span className="font-medium">Number of Features:</span>{' '}
                    {result.model_info.n_features_in}
                  </div>
                  <div>
                    <span className="font-medium">L2 Norm of Coefficients:</span>{' '}
                    {result.model_info.coef_l2_norm.toFixed(4)}
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Coefficient Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="font-medium">Most Important Feature:</span>{' '}
                    {result.coefficient_analysis.max_coefficient.feature} (
                    {result.coefficient_analysis.max_coefficient.value.toFixed(4)})
                  </div>
                  <div>
                    <span className="font-medium">Least Important Feature:</span>{' '}
                    {result.coefficient_analysis.min_coefficient.feature} (
                    {result.coefficient_analysis.min_coefficient.value.toFixed(4)})
                  </div>
                  <div>
                    <span className="font-medium">Mean Absolute Coefficient:</span>{' '}
                    {result.coefficient_analysis.mean_coefficient_abs.toFixed(4)}
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

export default RidgeRegressionDemo;
