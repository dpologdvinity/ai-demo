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

interface SentimentAnalysisParams {
  model_type: string;
  confidence_threshold: number;
  neutral_threshold: number;
  use_custom_texts: boolean;
  custom_texts: string[];
}

interface SentimentPrediction {
  text: string;
  sentiment: string;
  confidence: number;
  compound: number;
  scores: {
    positive: number;
    negative: number;
    neutral: number;
  };
}

interface AnalysisResult {
  success: boolean;
  predictions: SentimentPrediction[];
  distribution: {
    positive: number;
    negative: number;
    neutral: number;
    positive_pct: number;
    negative_pct: number;
    neutral_pct: number;
  };
  top_positive: SentimentPrediction[];
  top_negative: SentimentPrediction[];
  metrics: {
    total_texts: number;
    avg_confidence: number;
    avg_compound_score: number;
    avg_positive_confidence: number;
    avg_negative_confidence: number;
    avg_neutral_confidence: number;
  };
  visualization_data: any;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  model_info: {
    model_type: string;
    analyzer: string;
    version: string;
    description: string;
  };
  error?: string;
}

export function SentimentAnalysisDemo() {
  const [parameters, setParameters] = useState<SentimentAnalysisParams>({
    model_type: 'vader',
    confidence_threshold: 0.5,
    neutral_threshold: 0.05,
    use_custom_texts: false,
    custom_texts: [],
  });

  const [result, setResult] = useState<AnalysisResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['sentiment-analysis-info'],
    queryFn: () => apiService.getAlgorithmInfo('nlp', 'sentiment-analysis'),
  });

  // Analysis mutation
  const analyzeMutation = useMutation({
    mutationFn: (params: SentimentAnalysisParams) =>
      apiService.trainAlgorithm('nlp', 'sentiment-analysis', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleAnalyze = () => {
    analyzeMutation.mutate(parameters);
  };

  const handleParameterChange = (
    name: keyof SentimentAnalysisParams,
    value: any
  ) => {
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
      title="Sentiment Analysis"
      description={algorithmInfo?.metadata?.description}
      category="Natural Language Processing"
      sections={{
        parameters: (
          <div className="space-y-4">
            <Controls
              parameters={parameters}
              onChange={handleParameterChange}
              disabled={analyzeMutation.isPending}
            />
            <Button
              onClick={handleAnalyze}
              disabled={analyzeMutation.isPending}
              className="w-full"
            >
              {analyzeMutation.isPending ? 'Analyzing...' : 'Analyze Sentiment'}
            </Button>
          </div>
        ),
        visualization: (
          <div className="space-y-4">
            {analyzeMutation.isError && (
              <ErrorDisplay
                error={
                  analyzeMutation.error instanceof Error
                    ? analyzeMutation.error.message
                    : 'Analysis failed'
                }
              />
            )}
            {analyzeMutation.isPending && (
              <div className="flex items-center justify-center h-96">
                <LoadingSpinner />
              </div>
            )}
            {result && !analyzeMutation.isPending && (
              <Visualization result={result} />
            )}
            {!result && !analyzeMutation.isPending && !analyzeMutation.isError && (
              <div className="flex items-center justify-center h-96 text-muted-foreground">
                Run analysis to see sentiment predictions
              </div>
            )}
          </div>
        ),
        results: result && (
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Analysis Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Total Texts</p>
                    <p className="text-2xl font-bold">{result.metrics.total_texts}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Avg Confidence</p>
                    <p className="text-2xl font-bold">
                      {result.metrics.avg_confidence.toFixed(4)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">
                      Avg Compound Score
                    </p>
                    <p className="text-2xl font-bold">
                      {result.metrics.avg_compound_score.toFixed(4)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">
                      Execution Time
                    </p>
                    <p className="text-2xl font-bold">
                      {result.execution_time_ms.toFixed(2)}ms
                    </p>
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t space-y-2">
                  <h4 className="text-sm font-semibold">
                    Average Confidence by Sentiment
                  </h4>
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <div>
                      <span className="text-green-600 font-medium">Positive: </span>
                      {result.metrics.avg_positive_confidence.toFixed(4)}
                    </div>
                    <div>
                      <span className="text-red-600 font-medium">Negative: </span>
                      {result.metrics.avg_negative_confidence.toFixed(4)}
                    </div>
                    <div>
                      <span className="text-gray-600 font-medium">Neutral: </span>
                      {result.metrics.avg_neutral_confidence.toFixed(4)}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Sentiment Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-green-600">
                      Positive
                    </span>
                    <span className="text-sm">
                      {result.distribution.positive} (
                      {result.distribution.positive_pct.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${result.distribution.positive_pct}%` }}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-red-600">
                      Negative
                    </span>
                    <span className="text-sm">
                      {result.distribution.negative} (
                      {result.distribution.negative_pct.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-red-500 h-2 rounded-full"
                      style={{ width: `${result.distribution.negative_pct}%` }}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-600">
                      Neutral
                    </span>
                    <span className="text-sm">
                      {result.distribution.neutral} (
                      {result.distribution.neutral_pct.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-gray-500 h-2 rounded-full"
                      style={{ width: `${result.distribution.neutral_pct}%` }}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Model Information</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="font-medium">Model Type:</span>{' '}
                    {result.model_info.model_type.toUpperCase()}
                  </div>
                  <div>
                    <span className="font-medium">Analyzer:</span>{' '}
                    {result.model_info.analyzer}
                  </div>
                  <div>
                    <span className="font-medium">Version:</span>{' '}
                    {result.model_info.version}
                  </div>
                  <div className="pt-2 text-muted-foreground">
                    {result.model_info.description}
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

export default SentimentAnalysisDemo;
