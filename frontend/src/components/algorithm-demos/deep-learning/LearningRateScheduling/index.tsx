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

interface LearningRateParams {
  schedule_type: string;
  initial_lr: number;
  step_size: number;
  gamma: number;
  epochs: number;
  min_lr: number;
  max_lr: number;
  patience: number;
  random_state: number;
}

interface TrainingResult {
  success: boolean;
  schedule_data: Record<string, number[]>;
  loss_data: Record<string, number[]>;
  convergence_metrics: Record<string, Record<string, any>>;
  comparison_table: Array<Record<string, any>>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  error?: string;
}

export function LearningRateSchedulingDemo() {
  const [parameters, setParameters] = useState<LearningRateParams>({
    schedule_type: 'step',
    initial_lr: 0.1,
    step_size: 10,
    gamma: 0.1,
    epochs: 100,
    min_lr: 0.0001,
    max_lr: 0.5,
    patience: 5,
    random_state: 42,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['learning-rate-scheduling-info'],
    queryFn: () => apiService.getAlgorithmInfo('deep-learning', 'learning-rate-scheduling'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: LearningRateParams) =>
      apiService.trainAlgorithm('deep-learning', 'learning-rate-scheduling', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof LearningRateParams, value: any) => {
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
      title="Learning Rate Scheduling"
      description="Adjust learning rates during training for improved convergence"
      category="Deep Learning"
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
                onChange={handleParameterChange as (name: string, value: any) => void}
                algorithmInfo={algorithmInfo}
              />
              <Button
                onClick={handleTrain}
                disabled={trainMutation.isPending}
                className="w-full"
              >
                {trainMutation.isPending ? (
                  <>
                    <LoadingSpinner size="sm" className="mr-2" />
                    Training...
                  </>
                ) : (
                  'Train Model'
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
                    {result.comparison_table && result.comparison_table[0] && (
                      <>
                        <p>
                          <span className="font-medium">Best Schedule:</span>{' '}
                          {result.comparison_table[0].schedule}
                        </p>
                        <p>
                          <span className="font-medium">Final Loss:</span>{' '}
                          {result.comparison_table[0].final_loss?.toFixed(4)}
                        </p>
                      </>
                    )}
                    <p>
                      <span className="font-medium">Execution Time:</span>{' '}
                      {result.execution_time_ms.toFixed(2)} ms
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="mt-6">
            <Documentation />
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
                      Configure parameters and click "Train Model" to compare learning rate schedules
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

export default LearningRateSchedulingDemo;
