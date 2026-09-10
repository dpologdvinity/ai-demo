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

interface GloVeParams {
  embedding_dim: number;
  top_k: number;
  query_word: string;
  analogy_word_a: string;
  analogy_word_b: string;
  analogy_word_c: string;
}

interface TrainingResult {
  success: boolean;
  metrics: Record<string, any>;
  embeddings_2d: Array<{ word: string; x: number; y: number }>;
  similar_words: Array<{ word: string; similarity: number }>;
  analogy_result?: {
    query: string;
    result_word: string;
    similarity: number;
    top_results: Array<{ word: string; similarity: number }>;
  };
  cosine_similarity_matrix: Record<string, any>;
  vocabulary_sample: string[];
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  model_info: Record<string, any>;
  error?: string;
}

export function GloVeDemo() {
  const [parameters, setParameters] = useState<GloVeParams>({
    embedding_dim: 100,
    top_k: 10,
    query_word: 'king',
    analogy_word_a: 'king',
    analogy_word_b: 'man',
    analogy_word_c: 'woman',
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['glove-info'],
    queryFn: () => apiService.getAlgorithmInfo('nlp', 'glove'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: GloVeParams) =>
      apiService.trainAlgorithm('nlp', 'glove', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof GloVeParams, value: any) => {
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
      title="GloVe (Global Vectors for Word Representation)"
      description="Learn word embeddings that capture semantic and syntactic relationships"
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
                    Running GloVe...
                  </>
                ) : (
                  'Run GloVe'
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
                      <span className="font-medium">Embedding Dimension:</span>{' '}
                      {result.parameters_used.embedding_dim}
                    </p>
                    <p>
                      <span className="font-medium">Query Word:</span> {parameters.query_word}
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
                      Configure parameters and click "Run GloVe" to explore word embeddings
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

export default GloVeDemo;
