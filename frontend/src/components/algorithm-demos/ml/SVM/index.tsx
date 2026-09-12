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

interface SVMParams {
  C: number;
  kernel: string;
  gamma: string;
  degree: number;
}

interface SVMMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: number[][];
}

interface SVMModelInfo {
  C: number;
  kernel: string;
  gamma: string | number;
  degree: number;
  n_support_vectors: number;
  support_vectors_per_class: number[];
  n_features: number;
  n_classes: number;
  classes: number[];
}

interface VisualizationData {
  training_data: Array<{
    x: number;
    y: number;
    label: number;
    predicted: number;
    is_support_vector: boolean;
    type: string;
  }>;
  test_data: Array<{
    x: number;
    y: number;
    label: number;
    predicted: number;
    is_support_vector: boolean;
    type: string;
  }>;
  support_vectors: Array<{
    x: number;
    y: number;
    is_support_vector: boolean;
  }>;
  decision_boundary: any;
  feature_names: string[];
  target_names: string[];
}

interface TrainingResult {
  metrics: SVMMetrics;
  model_info: SVMModelInfo;
  visualization_data: VisualizationData;
  execution_time_ms: number;
  error?: string;
}

export function SVMDemo() {
  const [parameters, setParameters] = useState<SVMParams>({
    C: 1.0,
    kernel: 'rbf',
    gamma: 'scale',
    degree: 3,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['svm-info'],
    queryFn: () => apiService.getAlgorithmInfo('ml', 'svm'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: SVMParams) =>
      apiService.trainAlgorithm('ml', 'svm', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof SVMParams, value: any) => {
    setParameters((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  if (isLoadingInfo) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <AlgorithmLayout
      title="Support Vector Machine"
      description={algorithmInfo?.metadata?.description}
      sections={{
        parameters: (
          <div className="space-y-4">
            <Controls
              parameters={parameters}
              onChange={handleParameterChange}
              disabled={trainMutation.isPending}
            />
            <Button
              onClick={handleTrain}
              disabled={trainMutation.isPending}
              className="w-full"
            >
              {trainMutation.isPending ? 'Training...' : 'Train Model'}
            </Button>
          </div>
        ),
        visualization: (
          <div className="space-y-4">
            {trainMutation.isError && (
              <ErrorDisplay
                error={
                  trainMutation.error instanceof Error
                    ? trainMutation.error.message
                    : 'Training failed'
                }
              />
            )}
            {trainMutation.isPending && (
              <div className="flex items-center justify-center h-96">
                <LoadingSpinner />
              </div>
            )}
            {result && !trainMutation.isPending && (
              <Visualization result={result} />
            )}
            {!result && !trainMutation.isPending && !trainMutation.isError && (
              <div className="flex items-center justify-center h-96 text-muted-foreground">
                Train the model to see visualization
              </div>
            )}
          </div>
        ),
        results: result && (
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Accuracy</p>
                    <p className="text-2xl font-bold">
                      {(result.metrics.accuracy * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Precision</p>
                    <p className="text-2xl font-bold">
                      {(result.metrics.precision * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Recall</p>
                    <p className="text-2xl font-bold">
                      {(result.metrics.recall * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">F1 Score</p>
                    <p className="text-2xl font-bold">
                      {(result.metrics.f1_score * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t">
                  <p className="text-sm text-muted-foreground">
                    Execution Time: {result.execution_time_ms.toFixed(2)}ms
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Model Information</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="font-medium">Kernel:</span>{' '}
                    {result.model_info.kernel.toUpperCase()}
                  </div>
                  <div>
                    <span className="font-medium">C (Regularization):</span>{' '}
                    {result.model_info.C}
                  </div>
                  <div>
                    <span className="font-medium">Gamma:</span>{' '}
                    {result.model_info.gamma}
                  </div>
                  {result.model_info.kernel === 'poly' && (
                    <div>
                      <span className="font-medium">Polynomial Degree:</span>{' '}
                      {result.model_info.degree}
                    </div>
                  )}
                  <div>
                    <span className="font-medium">Support Vectors:</span>{' '}
                    {result.model_info.n_support_vectors} total
                  </div>
                  <div>
                    <span className="font-medium">Per Class:</span>
                    <div className="mt-1">
                      {result.model_info.support_vectors_per_class.map((count, idx) => (
                        <div key={idx} className="text-xs text-muted-foreground">
                          Class {result.model_info.classes[idx]}: {count} vectors
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Confusion Matrix</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr>
                        <th className="border p-2">Actual \ Predicted</th>
                        {result.visualization_data.target_names.map((name, idx) => (
                          <th key={idx} className="border p-2">{name}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.metrics.confusion_matrix.map((row, rowIdx) => (
                        <tr key={rowIdx}>
                          <th className="border p-2 font-medium">
                            {result.visualization_data.target_names[rowIdx]}
                          </th>
                          {row.map((value, colIdx) => (
                            <td
                              key={colIdx}
                              className={`border p-2 text-center ${
                                rowIdx === colIdx
                                  ? 'bg-green-100 dark:bg-green-900'
                                  : 'bg-red-50 dark:bg-red-950'
                              }`}
                            >
                              {value}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        ),
      }}
    >
      <Documentation metadata={algorithmInfo?.metadata} />
    </AlgorithmLayout>
  );
}

export default SVMDemo;
