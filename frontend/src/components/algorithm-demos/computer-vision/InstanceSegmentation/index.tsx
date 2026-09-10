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

interface InstanceSegmentationParams {
  confidence_threshold: number;
  model_backbone: string;
  mask_threshold: number;
  max_instances: number;
  image_index: number;
  nms_threshold: number;
}

interface TrainingResult {
  success: boolean;
  instances: any[];
  statistics: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
}

export function InstanceSegmentationDemo() {
  const [parameters, setParameters] = useState<InstanceSegmentationParams>({
    confidence_threshold: 0.5,
    model_backbone: 'resnet50',
    mask_threshold: 0.5,
    max_instances: 100,
    image_index: 0,
    nms_threshold: 0.5,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['instance-segmentation-info'],
    queryFn: () => apiService.getAlgorithmInfo('computer-vision', 'instance-segmentation'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: InstanceSegmentationParams) =>
      apiService.trainAlgorithm('computer-vision', 'instance-segmentation', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof InstanceSegmentationParams, value: any) => {
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
      title="Instance Segmentation (Mask R-CNN)"
      description="Detect and segment individual object instances with pixel-level masks"
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
                    Segmenting...
                  </>
                ) : (
                  'Segment Image'
                )}
              </Button>

              {trainMutation.error && (
                <ErrorDisplay
                  error={trainMutation.error}
                  title="Segmentation Error"
                />
              )}

              {result && result.success && (
                <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                    Results
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Total Instances:</span>{' '}
                      {result.statistics.total_instances}
                    </p>
                    <p>
                      <span className="font-medium">Unique Classes:</span>{' '}
                      {result.statistics.unique_classes}
                    </p>
                    <p>
                      <span className="font-medium">Coverage:</span>{' '}
                      {result.statistics.coverage_percentage.toFixed(1)}%
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
              <CardTitle>Segmentation Result</CardTitle>
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
                      Configure parameters and click "Segment Image" to detect and segment objects
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

export default InstanceSegmentationDemo;
