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

interface StyleTransferParams {
  content_image_index: number;
  style_image_index: number;
  iterations: number;
  content_weight: number;
  style_weight: number;
  learning_rate: number;
  image_size: number;
}

interface LossHistory {
  iteration: number;
  total_loss: number;
  content_loss: number;
  style_loss: number;
}

interface TrainingResult {
  success: boolean;
  statistics: Record<string, any>;
  visualization_data: Record<string, any>;
  loss_history: LossHistory[];
  feature_visualizations?: any[];
  execution_time_ms: number;
  model_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
}

export function StyleTransferDemo() {
  const [parameters, setParameters] = useState<StyleTransferParams>({
    content_image_index: 0,
    style_image_index: 0,
    iterations: 300,
    content_weight: 1.0,
    style_weight: 1000000.0,
    learning_rate: 0.003,
    image_size: 512,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['style-transfer-info'],
    queryFn: () => apiService.getAlgorithmInfo('computer-vision', 'style-transfer'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: StyleTransferParams) =>
      apiService.trainAlgorithm('computer-vision', 'style-transfer', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof StyleTransferParams, value: any) => {
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
      title="Neural Style Transfer"
      description="Apply artistic style from one image to the content of another using CNNs"
      category="Computer Vision"
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
                    Stylizing...
                  </>
                ) : (
                  'Apply Style'
                )}
              </Button>

              {trainMutation.error && (
                <ErrorDisplay
                  error={trainMutation.error}
                  title="Style Transfer Error"
                />
              )}

              {result && result.success && (
                <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                    Results
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Total Iterations:</span>{' '}
                      {result.statistics.total_iterations}
                    </p>
                    <p>
                      <span className="font-medium">Loss Reduction:</span>{' '}
                      {result.statistics.loss_reduction.toFixed(1)}%
                    </p>
                    <p>
                      <span className="font-medium">Final Loss:</span>{' '}
                      {result.statistics.final_total_loss.toFixed(4)}
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
              <CardTitle>Stylization Result</CardTitle>
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
                      Select content and style images, configure parameters, then click "Apply Style"
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

export default StyleTransferDemo;
