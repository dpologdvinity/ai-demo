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

interface IsolationForestParams {
  n_estimators: number;
  contamination: number;
  max_samples: string | number;
  random_state: number;
  n_samples: number;
  n_outliers_ratio: number;
}

interface AnomalyInfo {
  n_anomalies: number;
  n_normal: number;
  anomaly_ratio: number;
}

interface IsolationForestMetrics {
  n_anomalies: number;
  n_normal: number;
  anomaly_ratio: number;
  score_mean: number;
  score_std: number;
  score_min: number;
  score_max: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  accuracy?: number;
  confusion_matrix?: number[][];
}

interface VisualizationData {
  normal_points: Array<{
    x: number;
    y: number;
    score: number;
    prediction: number;
    true_label?: number;
  }>;
  anomaly_points: Array<{
    x: number;
    y: number;
    score: number;
    prediction: number;
    true_label?: number;
  }>;
  anomaly_scores: Array<{
    index: number;
    score: number;
    prediction: number;
  }>;
  score_stats: {
    min: number;
    max: number;
    mean: number;
    median: number;
    std: number;
  };
  n_normal: number;
  n_anomalies: number;
}

interface TrainingResult {
  success: boolean;
  metrics: IsolationForestMetrics;
  anomaly_info: AnomalyInfo;
  visualization_data: VisualizationData;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  error?: string;
}

export function IsolationForestDemo() {
  const [parameters, setParameters] = useState<IsolationForestParams>({
    n_estimators: 100,
    contamination: 0.1,
    max_samples: 'auto',
    random_state: 42,
    n_samples: 300,
    n_outliers_ratio: 0.1,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['isolation-forest-info'],
    queryFn: () => apiService.getAlgorithmInfo('ml', 'isolation-forest'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: IsolationForestParams) =>
      apiService.trainAlgorithm('ml', 'isolation-forest', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof IsolationForestParams, value: any) => {
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
      title="Isolation Forest"
      description={algorithmInfo?.metadata?.description}
      category="Machine Learning"
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
                <CardTitle>Anomaly Detection Results</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Anomalies Detected</p>
                    <p className="text-2xl font-bold text-red-600">
                      {result.anomaly_info.n_anomalies}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Normal Points</p>
                    <p className="text-2xl font-bold text-blue-600">
                      {result.anomaly_info.n_normal}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Anomaly Ratio</p>
                    <p className="text-2xl font-bold">
                      {(result.anomaly_info.anomaly_ratio * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Expected Contamination</p>
                    <p className="text-2xl font-bold">
                      {(parameters.contamination * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>

                {result.metrics.accuracy !== undefined && (
                  <div className="mt-4 pt-4 border-t">
                    <h4 className="font-semibold mb-3">Performance vs Ground Truth</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-muted-foreground">Accuracy</p>
                        <p className="text-lg font-bold">
                          {(result.metrics.accuracy * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Precision</p>
                        <p className="text-lg font-bold">
                          {((result.metrics.precision ?? 0) * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Recall</p>
                        <p className="text-lg font-bold">
                          {((result.metrics.recall ?? 0) * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">F1 Score</p>
                        <p className="text-lg font-bold">
                          {((result.metrics.f1_score ?? 0) * 100).toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="mt-4 pt-4 border-t">
                  <h4 className="font-semibold mb-3">Anomaly Score Statistics</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Mean Score</p>
                      <p className="font-semibold">
                        {result.metrics.score_mean.toFixed(4)}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Std Dev</p>
                      <p className="font-semibold">
                        {result.metrics.score_std.toFixed(4)}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Min Score</p>
                      <p className="font-semibold">
                        {result.metrics.score_min.toFixed(4)}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Max Score</p>
                      <p className="font-semibold">
                        {result.metrics.score_max.toFixed(4)}
                      </p>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground mt-3">
                    Note: More negative scores indicate higher anomaly likelihood
                  </p>
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
                <CardTitle>Model Parameters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="font-medium">Number of Trees:</span>
                    <span>{result.parameters_used.n_estimators}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Contamination:</span>
                    <span>{(result.parameters_used.contamination * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Max Samples per Tree:</span>
                    <span>{result.parameters_used.max_samples}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Total Data Points:</span>
                    <span>{result.parameters_used.n_samples}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Injected Outlier Ratio:</span>
                    <span>{(result.parameters_used.n_outliers_ratio * 100).toFixed(1)}%</span>
                  </div>
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

export default IsolationForestDemo;
