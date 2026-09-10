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

interface EdgeDetectionParams {
  threshold1: number;
  threshold2: number;
  aperture_size: number;
  l2gradient: boolean;
  image_index: number;
}

interface EdgeDetectionResult {
  success: boolean;
  statistics: {
    edge_pixel_count: number;
    total_pixels: number;
    edge_density: number;
    image_dimensions: Record<string, number>;
    threshold_ratio: number;
  };
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  algorithm_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
  error?: string;
}

export function EdgeDetectionDemo() {
  const [parameters, setParameters] = useState<EdgeDetectionParams>({
    threshold1: 50,
    threshold2: 150,
    aperture_size: 3,
    l2gradient: false,
    image_index: 0,
  });

  const [result, setResult] = useState<EdgeDetectionResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['edge-detection-info'],
    queryFn: () => apiService.getAlgorithmInfo('computer-vision', 'edge-detection'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: EdgeDetectionParams) =>
      apiService.trainAlgorithm('computer-vision', 'edge-detection', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof EdgeDetectionParams, value: any) => {
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
      title="Edge Detection (Canny)"
      description="Detect edges in images using the Canny edge detector"
      category="Computer Vision"
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
                    Detecting Edges...
                  </>
                ) : (
                  'Run Edge Detection'
                )}
              </Button>

              {trainMutation.error && (
                <ErrorDisplay
                  error={trainMutation.error}
                  title="Detection Error"
                />
              )}

              {result && result.success && (
                <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                    Results
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Edge Pixels:</span> {result.statistics.edge_pixel_count}
                    </p>
                    <p>
                      <span className="font-medium">Edge Density:</span>{' '}
                      {result.statistics.edge_density.toFixed(2)}%
                    </p>
                    <p>
                      <span className="font-medium">Total Pixels:</span> {result.statistics.total_pixels}
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
                      Configure parameters and click "Run Edge Detection" to detect edges
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

export default EdgeDetectionDemo;
