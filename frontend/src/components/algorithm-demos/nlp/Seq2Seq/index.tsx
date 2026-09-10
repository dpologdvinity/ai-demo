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

interface Seq2SeqParams {
  task: string;
  hidden_size: number;
  num_layers: number;
  dropout: number;
  attention: boolean;
  epochs: number;
  learning_rate: number;
  teacher_forcing_ratio: number;
  language_pair: string;
  use_custom_pairs: boolean;
  custom_input_sequences: string[];
  custom_target_sequences: string[];
}

interface TrainingResult {
  success: boolean;
  task: string;
  sample_predictions: Array<{
    input: string;
    target: string;
    prediction: string;
    bleu_score: number;
  }>;
  training_history: Array<{
    epoch: number;
    loss: number;
    bleu: number;
  }>;
  metrics: Record<string, any>;
  vocabulary_info: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  error?: string;
}

export function Seq2SeqDemo() {
  const [parameters, setParameters] = useState<Seq2SeqParams>({
    task: 'translation',
    hidden_size: 256,
    num_layers: 2,
    dropout: 0.3,
    attention: true,
    epochs: 50,
    learning_rate: 0.001,
    teacher_forcing_ratio: 0.5,
    language_pair: 'en-fr',
    use_custom_pairs: false,
    custom_input_sequences: [],
    custom_target_sequences: [],
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['seq2seq-info'],
    queryFn: () => apiService.getAlgorithmInfo('nlp', 'seq2seq'),
  });

  const trainMutation = useMutation({
    mutationFn: (params: Seq2SeqParams) =>
      apiService.trainAlgorithm('nlp', 'seq2seq', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof Seq2SeqParams, value: any) => {
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
      title="Seq2Seq (Sequence-to-Sequence)"
      description="Encoder-decoder architecture for sequence transformation tasks like machine translation"
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
                    Training Seq2Seq...
                  </>
                ) : (
                  'Train Model'
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
                      <span className="font-medium">Avg BLEU Score:</span>{' '}
                      {result.metrics.avg_bleu?.toFixed(4) || 'N/A'}
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
                      <span className="font-medium">Task:</span> {result.task}
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
                      Configure parameters and click "Train Model" to start training
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

export default Seq2SeqDemo;
