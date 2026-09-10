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

interface ConvParams {
  num_filters: number;
  kernel_size: number;
  stride: number;
  padding: string;
  activation: string;
  random_state: number;
}

interface ConvResult {
  success: boolean;
  input_image: number[][];
  filter_kernels: any[];
  feature_maps: any[];
  common_filters: Record<string, any>;
  output_dimensions: Record<string, any>;
  activation_stats: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
  parameters_used: Record<string, any>;
  error?: string;
}

export function ConvolutionalLayersDemo() {
  const [parameters, setParameters] = useState<ConvParams>({
    num_filters: 32,
    kernel_size: 3,
    stride: 1,
    padding: 'same',
    activation: 'relu',
    random_state: 42,
  });

  const [result, setResult] = useState<ConvResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['convolutional-layers-info'],
    queryFn: () => apiService.getAlgorithmInfo('deep-learning', 'convolutional-layers'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: ConvParams) =>
      apiService.trainAlgorithm('deep-learning', 'convolutional-layers', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof ConvParams, value: any) => {
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
      title="Convolutional Layers"
      description="Learn how convolution operations extract spatial features from images"
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
                    Computing...
                  </>
                ) : (
                  'Demonstrate Convolution'
                )}
              </Button>

              {trainMutation.error && (
                <ErrorDisplay
                  error={trainMutation.error}
                  title="Error"
                />
              )}

              {result && result.success && (
                <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                    Output Shape
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Height:</span> {result.output_dimensions?.height}
                    </p>
                    <p>
                      <span className="font-medium">Width:</span> {result.output_dimensions?.width}
                    </p>
                    <p>
                      <span className="font-medium">Channels:</span> {result.output_dimensions?.channels}
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
                      Configure parameters and click "Demonstrate Convolution" to see feature extraction
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

export default ConvolutionalLayersDemo;
