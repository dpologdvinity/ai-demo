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

interface DataAugParams {
  augmentation_types: string[];
  rotation_range: number;
  brightness_factor: number;
  num_augmented: number;
  random_state: number;
}

interface AugmentedImage {
  index: number;
  image_data: number[][][];
  augmentations_applied: string[];
  description: string;
}

interface AugmentationResult {
  success: boolean;
  original_image: number[][][];
  original_shape: number[];
  augmented_images: AugmentedImage[];
  augmentation_pipeline: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  error?: string;
}

export function DataAugmentationDemo() {
  const [parameters, setParameters] = useState<DataAugParams>({
    augmentation_types: ['rotation', 'flip_horizontal', 'brightness'],
    rotation_range: 30.0,
    brightness_factor: 0.3,
    num_augmented: 9,
    random_state: 42,
  });

  const [result, setResult] = useState<AugmentationResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['data-augmentation-info'],
    queryFn: () => apiService.getAlgorithmInfo('deep-learning', 'data-augmentation'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: DataAugParams) =>
      apiService.trainAlgorithm('deep-learning', 'data-augmentation', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof DataAugParams, value: any) => {
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
      title="Data Augmentation"
      description="Generate diverse training samples through image transformations"
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
                    Augmenting...
                  </>
                ) : (
                  'Generate Augmented Images'
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
                    Results
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Augmented Images:</span> {result.augmented_images?.length}
                    </p>
                    <p>
                      <span className="font-medium">Execution Time:</span> {result.execution_time_ms.toFixed(2)} ms
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
                      Configure parameters and click "Generate Augmented Images" to see augmentation effects
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

export default DataAugmentationDemo;
