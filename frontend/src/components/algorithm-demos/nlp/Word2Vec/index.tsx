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

interface Word2VecParams {
  vector_size: number;
  window: number;
  min_count: number;
  sg: number;
  epochs: number;
  use_custom_corpus: boolean;
  custom_corpus?: string;
  random_state: number;
}

interface TrainingResult {
  success: boolean;
  metrics: Record<string, any>;
  embeddings_2d: Array<{ word: string; x: number; y: number }>;
  similar_words: Record<string, Array<{ word: string; similarity: number }>>;
  analogies: Array<{ query: string; result: string; similarity: number }>;
  vocabulary_sample: string[];
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  model_info: Record<string, any>;
  error?: string;
}

export function Word2VecDemo() {
  const [parameters, setParameters] = useState<Word2VecParams>({
    vector_size: 100,
    window: 5,
    min_count: 5,
    sg: 0,
    epochs: 10,
    use_custom_corpus: false,
    random_state: 42,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['word2vec-info'],
    queryFn: () => apiService.getAlgorithmInfo('nlp', 'word2vec'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: Word2VecParams) =>
      apiService.trainAlgorithm('nlp', 'word2vec', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof Word2VecParams, value: any) => {
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
      title="Word2Vec"
      description="Learn word embeddings from large text corpora using neural networks"
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
                    Training Word2Vec...
                  </>
                ) : (
                  'Train Word2Vec'
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
                      <span className="font-medium">Vocabulary Size:</span>{' '}
                      {result.metrics.vocab_size}
                    </p>
                    <p>
                      <span className="font-medium">Vector Dimension:</span>{' '}
                      {result.parameters_used.vector_size}
                    </p>
                    <p>
                      <span className="font-medium">Algorithm:</span>{' '}
                      {result.parameters_used.sg === 0 ? 'CBOW' : 'Skip-gram'}
                    </p>
                    <p>
                      <span className="font-medium">Training Time:</span>{' '}
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
                      Configure parameters and click "Train Word2Vec" to learn word embeddings
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

export default Word2VecDemo;
