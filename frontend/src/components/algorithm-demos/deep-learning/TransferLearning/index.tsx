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

interface TransferLearningParams {
  base_model: string;
  strategy: string;
  freeze_layers: string;
  learning_rate: number;
  epochs: number;
  batch_size: number;
  dataset: string;
  random_state: number;
}

interface LayerInfo {
  name: string;
  trainable: boolean;
  num_params: number;
  frozen: boolean;
}

interface TrainingResult {
  success: boolean;
  metrics: Record<string, number>;
  training_history: Record<string, number[]>;
  confusion_matrix: number[][];
  layer_info: LayerInfo[];
  strategy_comparison?: Record<string, unknown>[];
  feature_maps?: Record<string, unknown>;
  sample_predictions: Record<string, unknown>[];
  visualization_data: Record<string, unknown>;
  execution_time_ms: number;
  model_info: Record<string, unknown>;
  parameters_used: Record<string, unknown>;
}

export function TransferLearningDemo() {
  const [parameters, setParameters] = useState<TransferLearningParams>({
    base_model: 'resnet18',
    strategy: 'fine_tune',
    freeze_layers: 'auto',
    learning_rate: 0.001,
    epochs: 10,
    batch_size: 16,
    dataset: 'flowers',
    random_state: 42,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['transfer-learning-info'],
    queryFn: () => apiService.getAlgorithmInfo('deep-learning', 'transfer-learning'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: TransferLearningParams) =>
      apiService.trainAlgorithm('deep-learning', 'transfer-learning', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof TransferLearningParams, value: unknown) => {
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
      title="Transfer Learning"
      description="Fine-tuning pre-trained models for new tasks"
      category="Deep Learning"
      difficulty="Intermediate"
    >
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Controls
                parameters={parameters}
                onChange={handleParameterChange as (name: string, value: unknown) => void}
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
                    Training Model...
                  </>
                ) : (
                  'Start Training'
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
                    Training Complete
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Val Accuracy:</span>{' '}
                      {(result.metrics?.val_accuracy as number * 100)?.toFixed(2) || 'N/A'}%
                    </p>
                    <p>
                      <span className="font-medium">Test Accuracy:</span>{' '}
                      {(result.metrics?.test_accuracy as number * 100)?.toFixed(2) || 'N/A'}%
                    </p>
                    <p>
                      <span className="font-medium">Training Time:</span>{' '}
                      {(result.execution_time_ms / 1000).toFixed(2)} sec
                    </p>
                    <p>
                      <span className="font-medium">Model:</span>{' '}
                      {(result.model_info?.base_model as string) || 'resnet18'}
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

        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Training Results</CardTitle>
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
                      Configure parameters and click "Start Training" to fine-tune the model
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

export default TransferLearningDemo;
