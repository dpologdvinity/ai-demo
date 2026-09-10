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

interface TopicModelingParams {
  n_topics: number;
  max_iterations: number;
  alpha: string | number;
  beta: string | number;
  min_df: number;
  max_df: number;
  use_custom_documents: boolean;
  custom_documents?: string[];
}

interface Topic {
  topic_id: number;
  top_words: Array<{ word: string; weight: number }>;
  keywords: string;
}

interface TrainingResult {
  success: boolean;
  topics: Topic[];
  coherence_score?: number;
  perplexity?: number;
  execution_time_ms: number;
  error?: string;
}

export function TopicModelingDemo() {
  const [parameters, setParameters] = useState<TopicModelingParams>({
    n_topics: 5,
    max_iterations: 100,
    alpha: 'auto',
    beta: 'auto',
    min_df: 2,
    max_df: 0.95,
    use_custom_documents: false,
    custom_documents: [],
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['topic-modeling-info'],
    queryFn: () => apiService.getAlgorithmInfo('nlp', 'topic-modeling'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: TopicModelingParams) =>
      apiService.post<TrainingResult>('/api/nlp/topic-modeling/train', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof TopicModelingParams, value: any) => {
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
      title="Topic Modeling (LDA)"
      description="Discover abstract topics in a document collection"
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
                onChange={handleParameterChange}
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
                    Modeling...
                  </>
                ) : (
                  'Discover Topics'
                )}
              </Button>

              {trainMutation.error && (
                <ErrorDisplay
                  error={trainMutation.error}
                  title="Modeling Error"
                />
              )}

              {result && result.success && (
                <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                    Results
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Topics Found:</span>{' '}
                      {result.topics.length}
                    </p>
                    {result.coherence_score !== undefined && (
                      <p>
                        <span className="font-medium">Coherence:</span>{' '}
                        {result.coherence_score.toFixed(4)}
                      </p>
                    )}
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
            <Documentation />
          </div>
        </div>

        {/* Visualization Panel */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Discovered Topics</CardTitle>
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
                      Configure parameters and click "Discover Topics" to see results
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

export default TopicModelingDemo;
