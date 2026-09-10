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

interface TFIDFParams {
  max_features: number;
  ngram_range: [number, number];
  min_df: number;
  max_df: number;
  use_idf: boolean;
  normalize: boolean;
  custom_documents?: string[];
}

interface TrainingResult {
  success: boolean;
  metrics: Record<string, any>;
  feature_names: string[];
  tfidf_matrix: number[][];
  top_terms_per_doc: any[];
  top_terms_global: Array<{ term: string; score: number }>;
  document_previews: string[];
  heatmap_data: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  model_info: Record<string, any>;
  error?: string;
}

export function TFIDFDemo() {
  const [parameters, setParameters] = useState<TFIDFParams>({
    max_features: 100,
    ngram_range: [1, 1],
    min_df: 1,
    max_df: 1.0,
    use_idf: true,
    normalize: true,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['tfidf-info'],
    queryFn: () => apiService.getAlgorithmInfo('nlp', 'tfidf'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: TFIDFParams) =>
      apiService.trainAlgorithm('nlp', 'tfidf', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof TFIDFParams, value: any) => {
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
      title="TF-IDF (Term Frequency-Inverse Document Frequency)"
      description="Measure term importance in documents using statistical techniques"
      category="Natural Language Processing"
      difficulty="Intermediate"
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
                    Running TF-IDF...
                  </>
                ) : (
                  'Run TF-IDF'
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
                      <span className="font-medium">Features:</span>{' '}
                      {result.feature_names.length}
                    </p>
                    <p>
                      <span className="font-medium">Documents:</span>{' '}
                      {result.tfidf_matrix.length}
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
                      Configure parameters and click "Run TF-IDF" to see term importance scores
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

export default TFIDFDemo;
