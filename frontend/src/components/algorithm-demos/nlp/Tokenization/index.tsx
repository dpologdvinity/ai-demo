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

interface TokenizationParams {
  tokenizer_type: string;
  lowercase: boolean;
  remove_punctuation: boolean;
  remove_stopwords: boolean;
  max_tokens: number;
  text_index: number;
  compare_mode: boolean;
  custom_text?: string;
}

interface TrainingResult {
  success: boolean;
  original_text: string;
  text_preview: string;
  main_result?: any;
  comparison_results: any[];
  statistics: Record<string, any>;
  frequency_distribution: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  error?: string;
}

export function TokenizationDemo() {
  const [parameters, setParameters] = useState<TokenizationParams>({
    tokenizer_type: 'word',
    lowercase: true,
    remove_punctuation: false,
    remove_stopwords: false,
    max_tokens: 100,
    text_index: 0,
    compare_mode: false,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['tokenization-info'],
    queryFn: () => apiService.getAlgorithmInfo('nlp', 'tokenization'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: TokenizationParams) =>
      apiService.trainAlgorithm('nlp', 'tokenization', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof TokenizationParams, value: any) => {
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
      title="Tokenization"
      description="Split text into individual tokens using different tokenization strategies"
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
                    Running Tokenization...
                  </>
                ) : (
                  'Tokenize Text'
                )}
              </Button>

              {trainMutation.error && (
                <ErrorDisplay
                  error={trainMutation.error}
                  title="Tokenization Error"
                />
              )}

              {result && result.success && (
                <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                    Results
                  </h3>
                  <div className="space-y-1 text-sm text-green-800 dark:text-green-200">
                    <p>
                      <span className="font-medium">Tokens:</span>{' '}
                      {result.statistics.token_count}
                    </p>
                    <p>
                      <span className="font-medium">Unique Tokens:</span>{' '}
                      {result.statistics.unique_token_count}
                    </p>
                    <p>
                      <span className="font-medium">Avg Token Length:</span>{' '}
                      {result.statistics.avg_token_length?.toFixed(2)}
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
                      Select a tokenizer and click "Tokenize Text" to see the results
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

export default TokenizationDemo;
