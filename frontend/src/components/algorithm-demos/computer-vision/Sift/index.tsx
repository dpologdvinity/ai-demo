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

interface SiftParams {
  nfeatures: number;
  nOctaveLayers: number;
  contrastThreshold: number;
  edgeThreshold: number;
  sigma: number;
  image_index: number;
  match_mode: boolean;
  match_image_index: number | null;
}

interface KeypointData {
  x: number;
  y: number;
  size: number;
  angle: number;
  response: number;
  octave: number;
}

interface SiftResult {
  success: boolean;
  statistics: {
    keypoint_count: number;
    image_dimensions: Record<string, number>;
    average_scale: number;
    average_response: number;
    scale_distribution: Record<string, number>;
    orientation_distribution: Record<string, number>;
    octave_distribution: Record<string, number>;
    descriptor_dimensions: number;
  };
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  algorithm_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
  keypoints: KeypointData[];
  match_statistics?: {
    match_count: number;
    total_matches: number;
    keypoints_image1: number;
    keypoints_image2: number;
    match_ratio: number;
    average_match_distance: number;
  };
  error?: string;
}

export function SiftDemo() {
  const [parameters, setParameters] = useState<SiftParams>({
    nfeatures: 500,
    nOctaveLayers: 3,
    contrastThreshold: 0.04,
    edgeThreshold: 10,
    sigma: 1.6,
    image_index: 0,
    match_mode: false,
    match_image_index: null,
  });

  const [result, setResult] = useState<SiftResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['sift-info'],
    queryFn: () => apiService.getAlgorithmInfo('computer-vision', 'sift'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: SiftParams) =>
      apiService.trainAlgorithm('computer-vision', 'sift', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof SiftParams, value: any) => {
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
      title="SIFT (Scale-Invariant Feature Transform)"
      description="Detect and describe scale and rotation-invariant features in images"
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
                    Detecting Features...
                  </>
                ) : (
                  'Run SIFT Detection'
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
                      <span className="font-medium">Keypoints Detected:</span>{' '}
                      {result.statistics.keypoint_count}
                    </p>
                    <p>
                      <span className="font-medium">Average Scale:</span>{' '}
                      {result.statistics.average_scale.toFixed(2)}
                    </p>
                    <p>
                      <span className="font-medium">Execution Time:</span>{' '}
                      {result.execution_time_ms.toFixed(2)} ms
                    </p>
                    {result.match_statistics && (
                      <p>
                        <span className="font-medium">Matches Found:</span>{' '}
                        {result.match_statistics.match_count}
                      </p>
                    )}
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
                      Configure parameters and click "Run SIFT Detection" to extract features
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

export default SiftDemo;
