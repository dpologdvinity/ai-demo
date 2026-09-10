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

interface VAEParams {
  latent_dim: number;
  encoder_hidden: number[];
  decoder_hidden: number[];
  learning_rate: number;
  epochs: number;
  beta: number;
  batch_size: number;
  random_state: number;
}

interface TrainingResult {
  success: boolean;
  metrics?: Record<string, number>;
  original_images?: number[][];
  reconstructed_images?: number[][];
  generated_images?: number[][];
  latent_space?: number[][];
  latent_labels?: number[];
  loss_history?: Record<string, number[]>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info?: Record<string, any>;
  parameters_used: Record<string, any>;
  error?: string;
}

export function VaeDemo() {
  const [parameters, setParameters] = useState<VAEParams>({
    latent_dim: 20,
    encoder_hidden: [128, 64],
    decoder_hidden: [64, 128],
    learning_rate: 0.001,
    epochs: 30,
    beta: 1.0,
    batch_size: 128,
    random_state: 42,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['vae-info'],
    queryFn: () => apiService.getAlgorithmInfo('deep-learning', 'vae'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: VAEParams) =>
      apiService.trainAlgorithm('deep-learning', 'vae', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof VAEParams, value: any) => {
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
      title="VAE (Variational Autoencoder)"
      description="Generative model learning a continuous latent space for data generation and interpolation"
      category="Deep Learning"
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
                    Training...
                  </>
                ) : (
                  'Train VAE'
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
                      <span className="font-medium">Final Loss:</span>{' '}
                      {result.metrics?.final_loss?.toFixed(2) || 'N/A'}
                    </p>
                    <p>
                      <span className="font-medium">Reconstruction Loss:</span>{' '}
                      {result.metrics?.final_reconstruction_loss?.toFixed(2) || 'N/A'}
                    </p>
                    <p>
                      <span className="font-medium">KL Divergence:</span>{' '}
                      {result.metrics?.final_kl_divergence?.toFixed(4) || 'N/A'}
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
                      Configure parameters and click "Train VAE" to start training
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

export default VaeDemo;
