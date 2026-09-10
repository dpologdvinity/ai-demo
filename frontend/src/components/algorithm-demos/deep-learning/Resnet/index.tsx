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

interface ResNetParams {
  model_variant: string;
  top_k: number;
  use_pretrained: boolean;
  image_index: number;
  random_state: number;
}

interface PredictionResult {
  class_id: number;
  class_name: string;
  confidence: number;
}

interface TrainingResult {
  success: boolean;
  predictions: PredictionResult[];
  input_image?: number[][][];
  input_shape: number[];
  feature_maps?: Record<string, unknown>;
  visualization_data: Record<string, unknown>;
  execution_time_ms: number;
  model_info: Record<string, unknown>;
  parameters_used: Record<string, unknown>;
}

export function ResnetDemo() {
  const [parameters, setParameters] = useState<ResNetParams>({
    model_variant: 'resnet18',
    top_k: 5,
    use_pretrained: true,
    image_index: 0,
    random_state: 42,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['resnet-info'],
    queryFn: () => apiService.getAlgorithmInfo('deep-learning', 'resnet'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: ResNetParams) =>
      apiService.trainAlgorithm('deep-learning', 'resnet', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof ResNetParams, value: unknown) => {
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
      title="ResNet (Residual Networks)"
      description="Deep convolutional neural networks with residual connections"
      category="Deep Learning"
      difficulty="Advanced"
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
                    Running Inference...
                  </>
                ) : (
                  'Run Inference'
                )}
              </Button>

              {trainMutation.error && (
                <ErrorDisplay
                  error={trainMutation.error}
                  title="Inference Error"
                />
              )}

              {result && result.success && (
                <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                    Results
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Top Prediction:</span>{' '}
                      {result.predictions[0]?.class_name}
                    </p>
                    <p>
                      <span className="font-medium">Confidence:</span>{' '}
                      {(result.predictions[0]?.confidence * 100).toFixed(2)}%
                    </p>
                    <p>
                      <span className="font-medium">Execution Time:</span>{' '}
                      {result.execution_time_ms.toFixed(2)} ms
                    </p>
                    <p>
                      <span className="font-medium">Model:</span>{' '}
                      {(result.model_info?.model_variant as string) || 'resnet18'}
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
              <CardTitle>Results</CardTitle>
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
                      Configure parameters and click "Run Inference" to classify an image
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

export default ResnetDemo;
