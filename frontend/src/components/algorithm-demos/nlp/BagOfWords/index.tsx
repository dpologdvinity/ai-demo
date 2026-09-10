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

interface BagOfWordsParams {
  max_features: number;
  ngram_range: [number, number];
  min_df: number;
  max_df: number;
  binary: boolean;
  custom_documents?: string[];
}

interface TrainingResult {
  success: boolean;
  metrics: Record<string, any>;
  vocabulary: string[];
  document_term_matrix: number[][];
  top_terms_per_doc: any[];
  most_frequent_terms: Array<{ term: string; frequency: number }>;
  least_frequent_terms: any[];
  document_previews: string[];
  heatmap_data: Record<string, any>;
  word_frequency_data: Record<string, any>;
  vocabulary_stats: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  model_info: Record<string, any>;
  error?: string;
}

export function BagOfWordsDemo() {
  const [parameters, setParameters] = useState<BagOfWordsParams>({
    max_features: 100,
    ngram_range: [1, 1],
    min_df: 1,
    max_df: 1.0,
    binary: false,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['bag-of-words-info'],
    queryFn: () => apiService.getAlgorithmInfo('nlp', 'bag-of-words'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: BagOfWordsParams) =>
      apiService.trainAlgorithm('nlp', 'bag-of-words', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof BagOfWordsParams, value: any) => {
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
      title="Bag of Words"
      description="Convert text documents into vector representations based on word frequencies"
      category="Natural Language Processing"
      difficulty="Beginner"
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
                    Running Bag of Words...
                  </>
                ) : (
                  'Run Bag of Words'
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
                      {result.vocabulary.length}
                    </p>
                    <p>
                      <span className="font-medium">Documents:</span>{' '}
                      {result.document_term_matrix.length}
                    </p>
                    <p>
                      <span className="font-medium">Sparsity:</span>{' '}
                      {result.metrics.sparsity?.toFixed(2)}%
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
                      Configure parameters and click "Run Bag of Words" to visualize word frequencies
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

export default BagOfWordsDemo;
