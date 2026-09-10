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

interface PoolingParams {
  pool_type: string;
  pool_size: number;
  stride: number;
  padding: number;
  input_size: number;
  num_channels: number;
  random_state: number;
}

interface PoolingResult {
  success: boolean;
  input_feature_map: number[][][];
  pooled_output: number[][][];
  max_positions: number[][][][];
  dimension_info: Record<string, any>;
  pooling_windows: any[];
  comparison_data: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  error?: string;
}

export function PoolingLayersDemo() {
  const [parameters, setParameters] = useState<PoolingParams>({
    pool_type: 'max',
    pool_size: 2,
    stride: 2,
    padding: 0,
    input_size: 8,
    num_channels: 1,
    random_state: 42,
  });

  const [result, setResult] = useState<PoolingResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['pooling-layers-info'],
    queryFn: () => apiService.getAlgorithmInfo('deep-learning', 'pooling-layers'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: PoolingParams) =>
      apiService.trainAlgorithm('deep-learning', 'pooling-layers', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof PoolingParams, value: any) => {
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
      title="Pooling Layers"
      description="Reduce spatial dimensions while preserving important features"
      category="Deep Learning"
      difficulty="Beginner"
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
                  'Apply Pooling'
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
                    Output Dimensions
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Input:</span> {result.dimension_info?.input_shape?.join(' × ')}
                    </p>
                    <p>
                      <span className="font-medium">Output:</span> {result.dimension_info?.output_shape?.join(' × ')}
                    </p>
                    <p>
                      <span className="font-medium">Reduction:</span> {result.dimension_info?.reduction_factor}x
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
                      Configure parameters and click "Apply Pooling" to see dimension reduction
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

export default PoolingLayersDemo;
