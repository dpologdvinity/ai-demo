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

interface BertParams {
  model_name: string;
  learning_rate: number;
  epochs: number;
  batch_size: number;
  max_length: number;
  use_custom_dataset: boolean;
  custom_texts: string[];
  custom_labels: number[];
}

interface TrainingResult {
  success: boolean;
  training_history: Array<{
    epoch: number;
    train_loss: number;
    train_accuracy: number;
    val_loss?: number;
    val_accuracy?: number;
  }>;
  predictions: Array<{
    text: string;
    predicted_label: number;
    predicted_class: string;
    confidence: number;
    probabilities: number[];
  }>;
  confusion_matrix?: {
    matrix: number[][];
    labels: string[];
    accuracy: number;
    precision: number[];
    recall: number[];
    f1_score: number[];
  };
  metrics: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  model_info: Record<string, any>;
  error?: string;
}

export function BertFinetuningDemo() {
  const [parameters, setParameters] = useState<BertParams>({
    model_name: 'bert-base-uncased',
    learning_rate: 2e-5,
    epochs: 3,
    batch_size: 16,
    max_length: 128,
    use_custom_dataset: false,
    custom_texts: [],
    custom_labels: [],
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['bert-finetuning-info'],
    queryFn: () => apiService.getAlgorithmInfo('nlp', 'bert-finetuning'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: BertParams) =>
      apiService.trainAlgorithm('nlp', 'bert-finetuning', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof BertParams, value: any) => {
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
      title="BERT Fine-tuning"
      description="Fine-tune pre-trained BERT model for text classification tasks"
      category="Natural Language Processing"
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
                    Fine-tuning BERT...
                  </>
                ) : (
                  'Fine-tune Model'
                )}
              </Button>

              {trainMutation.error && (
                <ErrorDisplay
                  error={trainMutation.error}
                  title="Training Error"
                />
              )}

              {result && result.success && (
                <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                    Results
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Final Accuracy:</span>{' '}
                      {(result.metrics.accuracy * 100).toFixed(2)}%
                    </p>
                    <p>
                      <span className="font-medium">Final Loss:</span>{' '}
                      {result.metrics.final_loss?.toFixed(4) || 'N/A'}
                    </p>
                    <p>
                      <span className="font-medium">Execution Time:</span>{' '}
                      {result.execution_time_ms.toFixed(2)} ms
                    </p>
                    <p>
                      <span className="font-medium">Epochs Trained:</span>{' '}
                      {result.training_history.length}
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
              <CardTitle>Results</CardTitle>
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
                      Configure parameters and click "Fine-tune Model" to train BERT
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

export default BertFinetuningDemo;
