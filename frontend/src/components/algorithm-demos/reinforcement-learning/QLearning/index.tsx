import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { AlgorithmLayout } from '@/components/common/AlgorithmLayout';
import Button from '@/components/common/Button';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { ErrorDisplay } from '@/components/common/ErrorDisplay';
import { apiService } from '@/services/api';
import { StepPlayer, useStepPlayback } from '@/components/common/StepPlayer';
import { MathPanel } from '@/components/common/MathPanel';
import { Controls } from './Controls';
import { Visualization } from './Visualization';
import { Documentation } from './Documentation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface QLearningParams {
  learning_rate: number;
  discount_factor: number;
  epsilon: number;
  episodes: number;
  grid_size: number;
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

export function QLearningDemo() {
  const [parameters, setParameters] = useState<QLearningParams>({
    learning_rate: 0.1,
    discount_factor: 0.99,
    epsilon: 0.1,
    episodes: 1000,
    grid_size: 5,
    random_state: 42,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Initialize step playback for episode replay (only if we have episode data)
  const episodeRewards = result?.success ? (result.visualization_data?.episode_rewards || []) : [];
  const totalEpisodes = episodeRewards.length;
  const stepPlayback = useStepPlayback(totalEpisodes > 0 ? totalEpisodes : 1);
  const currentReward = totalEpisodes > 0 ? episodeRewards[stepPlayback.currentStep] : 0;

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['q-learning-info'],
    queryFn: () => apiService.getAlgorithmInfo('reinforcement-learning', 'q-learning'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: QLearningParams) =>
      apiService.trainAlgorithm('reinforcement-learning', 'q-learning', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof QLearningParams, value: any) => {
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
      title="Q-Learning"
      description="Model-free reinforcement learning algorithm for discrete action and state spaces"
      category="Reinforcement Learning"
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
                    Training Q-Learning...
                  </>
                ) : (
                  'Train Q-Learning'
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
                      <span className="font-medium">Avg Reward (Last 100):</span>{' '}
                      {(result.metrics.avg_reward_last_100 || 0).toFixed(2)}
                    </p>
                    <p>
                      <span className="font-medium">Success Rate:</span>{' '}
                      {((result.metrics.success_rate || 0) * 100).toFixed(1)}%
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
        <div className="lg:col-span-2 space-y-6">
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
                      Configure parameters and click "Train Q-Learning" to start training
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Step Player - show only when result has episode data */}
          {result && result.success && totalEpisodes > 0 && (
            <div>
              <StepPlayer
                totalSteps={totalEpisodes}
                {...stepPlayback.playerProps}
              />
            </div>
          )}

          {/* Q-Learning Update Rule Formula */}
          {result && result.success && totalEpisodes > 0 && (
            <MathPanel
              title="Q-Learning Update Rule"
              formula="Q(s,a) \\leftarrow Q(s,a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s',a') - Q(s,a) \\right]"
              substitution={`r_{${stepPlayback.currentStep + 1}} = ${currentReward.toFixed(2)}`}
              note={`α = ${result.parameters_used.learning_rate}, γ = ${result.parameters_used.discount_factor}`}
            />
          )}
        </div>
      </div>
    </AlgorithmLayout>
  );
}

export default QLearningDemo;
