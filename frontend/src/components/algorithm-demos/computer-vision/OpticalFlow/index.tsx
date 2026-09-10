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

interface OpticalFlowParams {
  method: string;
  pyr_scale: number;
  levels: number;
  winsize: number;
  iterations: number;
  image_pair_index: number;
}

interface FlowVector {
  x: number;
  y: number;
  dx: number;
  dy: number;
  magnitude: number;
  angle: number;
}

interface OpticalFlowResult {
  success: boolean;
  statistics: {
    average_magnitude: number;
    max_magnitude: number;
    min_magnitude: number;
    median_magnitude: number;
    flow_coverage: number;
    primary_direction: number;
    image_dimensions: Record<string, number>;
    total_pixels: number;
  };
  visualization_data: Record<string, any>;
  flow_vectors: FlowVector[];
  execution_time_ms: number;
  algorithm_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
  error?: string;
}

export function OpticalFlowDemo() {
  const [parameters, setParameters] = useState<OpticalFlowParams>({
    method: 'farneback',
    pyr_scale: 0.5,
    levels: 3,
    winsize: 15,
    iterations: 3,
    image_pair_index: 0,
  });

  const [result, setResult] = useState<OpticalFlowResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['optical-flow-info'],
    queryFn: () => apiService.getAlgorithmInfo('computer-vision', 'optical-flow'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: OpticalFlowParams) =>
      apiService.trainAlgorithm('computer-vision', 'optical-flow', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof OpticalFlowParams, value: any) => {
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
      title="Optical Flow"
      description="Estimate motion between consecutive video frames"
      category="Computer Vision"
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
                    Computing Flow...
                  </>
                ) : (
                  'Run Optical Flow'
                )}
              </Button>

              {trainMutation.error && (
                <ErrorDisplay
                  error={trainMutation.error}
                  title="Computation Error"
                />
              )}

              {result && result.success && (
                <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                    Results
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Avg Magnitude:</span>{' '}
                      {result.statistics.average_magnitude.toFixed(2)}
                    </p>
                    <p>
                      <span className="font-medium">Max Magnitude:</span>{' '}
                      {result.statistics.max_magnitude.toFixed(2)}
                    </p>
                    <p>
                      <span className="font-medium">Flow Coverage:</span>{' '}
                      {result.statistics.flow_coverage.toFixed(2)}%
                    </p>
                    <p>
                      <span className="font-medium">Primary Direction:</span>{' '}
                      {result.statistics.primary_direction.toFixed(1)}°
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
                      Configure parameters and click "Run Optical Flow" to estimate motion
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

export default OpticalFlowDemo;
