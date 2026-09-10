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

interface AutoencoderParams {
  latent_dim: number;
  hidden_dim: number;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  random_state: number;
}

interface TrainingResult {
  reconstruction_loss: number;
  avg_pixel_error: number;
  loss_history: Array<{ epoch: number; train_loss: number; val_loss: number }>;
  original_images: number[][];
  reconstructed_images: number[][];
  latent_representations: Array<{ x: number; y: number; label: number }>;
  visualization_data: {
    n_samples: number;
    image_shape: number[];
    latent_dim: number;
    projection_method: string;
  };
  execution_time_ms: number;
  model_info: {
    latent_dim: number;
    hidden_dim: number;
    total_epochs: number;
    encoder_params: number;
    decoder_params: number;
  };
}

export function AutoencoderDemo() {
  const [parameters, setParameters] = useState<AutoencoderParams>({
    latent_dim: 32,
    hidden_dim: 128,
    epochs: 50,
    learning_rate: 0.001,
    batch_size: 64,
    random_state: 42,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['autoencoder-info'],
    queryFn: () => apiService.getAlgorithmInfo('deep-learning', 'autoencoder'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: AutoencoderParams) =>
      apiService.trainAlgorithm('deep-learning', 'autoencoder', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof AutoencoderParams, value: any) => {
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
      title="Autoencoder"
      description="Unsupervised learning architecture for dimensionality reduction and data compression"
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
                  'Train Autoencoder'
                )}
              </Button>

              {trainMutation.error && (
                <ErrorDisplay
                  error={trainMutation.error}
                  title="Training Error"
                />
              )}

              {result && (
                <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                    Results
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Reconstruction Loss:</span>{' '}
                      {result.reconstruction_loss.toFixed(6)}
                    </p>
                    <p>
                      <span className="font-medium">Avg Pixel Error:</span>{' '}
                      {result.avg_pixel_error.toFixed(6)}
                    </p>
                    <p>
                      <span className="font-medium">Execution Time:</span>{' '}
                      {result.execution_time_ms.toFixed(2)} ms
                    </p>
                    <p>
                      <span className="font-medium">Latent Dimension:</span>{' '}
                      {result.model_info.latent_dim}
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
              ) : result ? (
                <Visualization result={result} />
              ) : (
                <div className="flex items-center justify-center h-96 text-gray-500 dark:text-gray-400">
                  <div className="text-center">
                    <p className="text-lg font-medium mb-2">No Results Yet</p>
                    <p className="text-sm">
                      Configure parameters and click "Train Autoencoder" to start training
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

export default AutoencoderDemo;
