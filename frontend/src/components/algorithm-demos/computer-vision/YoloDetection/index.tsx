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

interface YoloDetectionParams {
  confidence_threshold: number;
  iou_threshold: number;
  model_version: string;
  max_detections: number;
  image_index: number;
  class_filter: string;
}

interface Detection {
  bbox: number[];
  class_name: string;
  class_id: number;
  confidence: number;
}

interface TrainingResult {
  success: boolean;
  detections: Detection[];
  statistics: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
}

export function YoloDetectionDemo() {
  const [parameters, setParameters] = useState<YoloDetectionParams>({
    confidence_threshold: 0.25,
    iou_threshold: 0.45,
    model_version: 'yolov8n',
    max_detections: 100,
    image_index: 0,
    class_filter: 'all',
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['yolo-detection-info'],
    queryFn: () => apiService.getAlgorithmInfo('computer-vision', 'yolo-detection'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: YoloDetectionParams) =>
      apiService.trainAlgorithm('computer-vision', 'yolo-detection', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof YoloDetectionParams, value: any) => {
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
      title="YOLO Object Detection"
      description="Real-time object detection with bounding boxes and class labels"
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
                    Detecting...
                  </>
                ) : (
                  'Detect Objects'
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
                    Detection Results
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Objects Detected:</span>{' '}
                      {result.statistics.total_detections}
                    </p>
                    <p>
                      <span className="font-medium">Unique Classes:</span>{' '}
                      {Object.keys(result.statistics.class_counts || {}).length}
                    </p>
                    <p>
                      <span className="font-medium">Avg Confidence:</span>{' '}
                      {(result.statistics.avg_confidence * 100).toFixed(1)}%
                    </p>
                    <p>
                      <span className="font-medium">Inference Time:</span>{' '}
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
              <CardTitle>Detection Results</CardTitle>
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
                      Configure parameters and click "Detect Objects" to run YOLO detection
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

export default YoloDetectionDemo;
