import React, { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { AlertCircle } from 'lucide-react';
import { apiService } from '@/services/api';
import { AlgorithmLayout } from '@/components/common/AlgorithmLayout';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { ErrorDisplay } from '@/components/common/ErrorDisplay';
import { Controls } from './Controls';
import { Visualization } from './Visualization';
import { Documentation } from './Documentation';

interface LogisticRegressionResult {
  success: boolean;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
  predictions: number[];
  visualization_data: {
    scatter_data: any;
    confusion_matrix: any;
    probability_data: any;
  };
  execution_time_ms: number;
  parameters_used: Record<string, any>;
}

export default function LogisticRegressionDemo() {
  const [parameters, setParameters] = useState({
    C: 1.0,
    penalty: 'l2',
    max_iter: 100,
    solver: 'lbfgs',
  });

  const [datasetName, setDatasetName] = useState('iris');

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: infoLoading } = useQuery({
    queryKey: ['logistic-regression-info'],
    queryFn: () => apiService.getAlgorithmInfo('ml', 'logistic-regression'),
  });

  // Train model mutation
  const trainMutation = useMutation({
    mutationFn: (params: any) =>
      apiService.trainAlgorithm('ml', 'logistic-regression', params),
    onError: (error: any) => {
      console.error('Training failed:', error);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate({
      parameters,
      dataset_name: datasetName,
      normalize: true,
    });
  };

  const handleParameterChange = (name: string, value: any) => {
    setParameters((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  if (infoLoading) {
    return <LoadingSpinner />;
  }

  const result = trainMutation.data as LogisticRegressionResult | undefined;

  return (
    <AlgorithmLayout
      title="Logistic Regression"
      description="Binary and multiclass classification using logistic function"
      difficulty={algorithmInfo?.metadata?.difficulty || 'Beginner'}
      complexity={algorithmInfo?.metadata?.complexity}
      tags={algorithmInfo?.metadata?.tags || []}
    >
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-1 space-y-6">
          <Controls
            parameters={parameters}
            onParameterChange={handleParameterChange}
            onTrain={handleTrain}
            isTraining={trainMutation.isPending}
            datasetName={datasetName}
            onDatasetChange={setDatasetName}
            availableDatasets={algorithmInfo?.available_datasets || ['iris']}
          />

          {trainMutation.isError && (
            <ErrorDisplay
              error={
                trainMutation.error instanceof Error
                  ? trainMutation.error.message
                  : 'Training failed'
              }
            />
          )}

          {result && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Metrics</h3>
              <div className="grid grid-cols-2 gap-3">
                <MetricCard label="Accuracy" value={result.metrics.accuracy} />
                <MetricCard label="Precision" value={result.metrics.precision} />
                <MetricCard label="Recall" value={result.metrics.recall} />
                <MetricCard label="F1 Score" value={result.metrics.f1_score} />
              </div>
              <p className="text-sm text-muted-foreground">
                Execution time: {result.execution_time_ms.toFixed(2)}ms
              </p>
            </div>
          )}
        </div>

        <div className="lg:col-span-2 space-y-6">
          {result && (
            <Visualization
              scatterData={result.visualization_data.scatter_data}
              confusionMatrix={result.visualization_data.confusion_matrix}
              probabilityData={result.visualization_data.probability_data}
            />
          )}

          {!result && !trainMutation.isPending && (
            <div className="flex items-center justify-center h-96 border-2 border-dashed border-border rounded-lg">
              <div className="text-center text-muted-foreground">
                <AlertCircle className="mx-auto h-12 w-12 mb-4 opacity-50" />
                <p className="text-lg font-medium">No results yet</p>
                <p className="text-sm">
                  Configure parameters and click Train Model to see results
                </p>
              </div>
            </div>
          )}

          {trainMutation.isPending && (
            <div className="flex items-center justify-center h-96">
              <LoadingSpinner />
            </div>
          )}

          <Documentation metadata={algorithmInfo?.metadata} />
        </div>
      </div>
    </AlgorithmLayout>
  );
}

function MetricCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-muted/50 rounded-lg p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-2xl font-bold">{(value * 100).toFixed(1)}%</p>
    </div>
  );
}
