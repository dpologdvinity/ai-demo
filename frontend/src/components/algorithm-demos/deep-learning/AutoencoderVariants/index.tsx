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

interface AutoencoderVariantsParams {
  variant: string;
  latent_dim: number;
  epochs: number;
  learning_rate: number;
  noise_factor: number;
  sparsity_weight: number;
  batch_size: number;
  random_state: number;
}

interface TrainingResult {
  success: boolean;
  variant: string;
  metrics: Record<string, number>;
  loss_history: Array<{ epoch: number; train_loss: number; val_loss: number }>;
  original_images: number[][];
  noisy_images?: number[][];
  reconstructed_images: number[][];
  latent_space: number[][];
  latent_labels: number[];
  reconstruction_errors: number[];
  learned_filters: number[][];
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
  parameters_used: Record<string, any>;
}

export function AutoencoderVariantsDemo() {
  const [parameters, setParameters] = useState<AutoencoderVariantsParams>({
    variant: 'vanilla',
    latent_dim: 32,
    epochs: 10,
    learning_rate: 0.001,
    noise_factor: 0.3,
    sparsity_weight: 0.001,
    batch_size: 128,
    random_state: 42,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['autoencoder-variants-info'],
    queryFn: () => apiService.getAlgorithmInfo('deep-learning', 'autoencoder-variants'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: AutoencoderVariantsParams) =>
      apiService.trainAlgorithm('deep-learning', 'autoencoder-variants', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof AutoencoderVariantsParams, value: any) => {
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
      title="Autoencoder Variants"
      description="Explore different autoencoder architectures: Vanilla, Denoising, Sparse, and Contractive"
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
                  'Train Variant'
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
                    Results ({result.variant})
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Final Loss:</span>{' '}
                      {result.metrics.final_loss?.toFixed(6) || 'N/A'}
                    </p>
                    <p>
                      <span className="font-medium">Reconstruction MSE:</span>{' '}
                      {result.metrics.reconstruction_mse?.toFixed(6) || 'N/A'}
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
                      Select a variant and click "Train Variant" to start training
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

export default AutoencoderVariantsDemo;
