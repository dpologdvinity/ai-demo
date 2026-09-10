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

interface PPOParams {
  learning_rate: number;
  gamma: number;
  clip_epsilon: number;
  epochs: number;
  episodes: number;
  gae_lambda: number;
  batch_size: number;
  hidden_size: number;
  random_state: number;
}

interface TrainingResult {
  success: boolean;
  metrics: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  error?: string;
}

export function PPODemo() {
  const [parameters, setParameters] = useState<PPOParams>({
    learning_rate: 0.0003,
    gamma: 0.99,
    clip_epsilon: 0.2,
    epochs: 4,
    episodes: 500,
    gae_lambda: 0.95,
    batch_size: 64,
    hidden_size: 128,
    random_state: 42,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['ppo-info'],
    queryFn: () => apiService.getAlgorithmInfo('reinforcement-learning', 'ppo'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: PPOParams) =>
      apiService.trainAlgorithm('reinforcement-learning', 'ppo', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof PPOParams, value: any) => {
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
      title="PPO (Proximal Policy Optimization)"
      description="Policy gradient method with clipped objective for stable and sample-efficient training"
      category="Reinforcement Learning"
      difficulty="Advanced"
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
                    Training PPO...
                  </>
                ) : (
                  'Train PPO Agent'
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
                      <span className="font-medium">Avg Reward (last 100):</span>{' '}
                      {result.metrics.avg_reward_last_100?.toFixed(2) ?? 'N/A'}
                    </p>
                    <p>
                      <span className="font-medium">Success Rate:</span>{' '}
                      {result.metrics.success_rate?.toFixed(3) ?? 'N/A'}
                    </p>
                    <p>
                      <span className="font-medium">Avg Clip Fraction:</span>{' '}
                      {result.metrics.avg_clip_fraction?.toFixed(4) ?? 'N/A'}
                    </p>
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
              <CardTitle>Training Progress</CardTitle>
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
                      Configure parameters and click "Train PPO Agent" to see training progress
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

export default PPODemo;
