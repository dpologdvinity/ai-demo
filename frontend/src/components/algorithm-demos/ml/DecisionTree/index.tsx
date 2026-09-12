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

interface DecisionTreeParams {
  max_depth: number | null;
  min_samples_split: number;
  min_samples_leaf: number;
  criterion: string;
  dataset_name: string;
  random_state: number;
}

interface TreeNode {
  id: number;
  type: string;
  class?: number;
  class_name?: string;
  samples: number;
  value?: number[];
  impurity: number;
  feature?: number;
  feature_name?: string;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
}

interface TrainingResult {
  success: boolean;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    n_nodes: number;
    n_leaves: number;
    max_depth_achieved: number;
  };
  predictions: {
    y_test: number[];
    y_pred: number[];
    y_pred_proba: number[][];
  };
  visualization_data: {
    confusion_matrix: number[][];
    labels: string[];
    tree_structure: TreeNode;
    tree_text: string;
    feature_importance: {
      features: string[];
      importance: number[];
    };
  };
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  error?: string;
}

export function DecisionTreeDemo() {
  const [parameters, setParameters] = useState<DecisionTreeParams>({
    max_depth: 5,
    min_samples_split: 2,
    min_samples_leaf: 1,
    criterion: 'gini',
    dataset_name: 'iris',
    random_state: 42,
  });

  const [result, setResult] = useState<TrainingResult | null>(null);

  // Fetch algorithm info
  const { data: algorithmInfo, isLoading: isLoadingInfo } = useQuery({
    queryKey: ['decision-tree-info'],
    queryFn: () => apiService.getAlgorithmInfo('ml', 'decision-tree'),
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: (params: DecisionTreeParams) =>
      apiService.trainAlgorithm('ml', 'decision-tree', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof DecisionTreeParams, value: any) => {
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
      title="Decision Tree Classifier"
      description={algorithmInfo?.metadata?.description}
      sections={{
        parameters: (
          <div className="space-y-4">
            <Controls
              parameters={parameters}
              onChange={handleParameterChange}
              disabled={trainMutation.isPending}
            />
            <Button
              onClick={handleTrain}
              disabled={trainMutation.isPending}
              className="w-full"
            >
              {trainMutation.isPending ? 'Training...' : 'Train Model'}
            </Button>
          </div>
        ),
        visualization: (
          <div className="space-y-4">
            {trainMutation.isError && (
              <ErrorDisplay
                error={
                  trainMutation.error instanceof Error
                    ? trainMutation.error.message
                    : 'Training failed'
                }
              />
            )}
            {trainMutation.isPending && (
              <div className="flex items-center justify-center h-96">
                <LoadingSpinner />
              </div>
            )}
            {result && !trainMutation.isPending && (
              <Visualization result={result} />
            )}
            {!result && !trainMutation.isPending && !trainMutation.isError && (
              <div className="flex items-center justify-center h-96 text-muted-foreground">
                Train the model to see visualization
              </div>
            )}
          </div>
        ),
        results: result && (
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Accuracy</p>
                    <p className="text-2xl font-bold">
                      {result.metrics.accuracy.toFixed(4)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">F1 Score</p>
                    <p className="text-2xl font-bold">
                      {result.metrics.f1_score.toFixed(4)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Precision</p>
                    <p className="text-2xl font-bold">
                      {result.metrics.precision.toFixed(4)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Recall</p>
                    <p className="text-2xl font-bold">
                      {result.metrics.recall.toFixed(4)}
                    </p>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t">
                  <p className="text-sm text-muted-foreground">
                    Execution Time: {result.execution_time_ms.toFixed(2)}ms
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Tree Structure</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <span className="font-medium">Total Nodes:</span>{' '}
                      {result.metrics.n_nodes}
                    </div>
                    <div>
                      <span className="font-medium">Leaf Nodes:</span>{' '}
                      {result.metrics.n_leaves}
                    </div>
                    <div>
                      <span className="font-medium">Max Depth:</span>{' '}
                      {result.metrics.max_depth_achieved}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Feature Importance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {result.visualization_data.feature_importance.features.map(
                    (feature, idx) => {
                      const importance =
                        result.visualization_data.feature_importance.importance[idx];
                      return (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-sm">
                            <span>{feature}</span>
                            <span className="font-medium">
                              {(importance * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div className="w-full bg-secondary rounded-full h-2">
                            <div
                              className="bg-primary h-2 rounded-full"
                              style={{ width: `${importance * 100}%` }}
                            />
                          </div>
                        </div>
                      );
                    }
                  )}
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

export default DecisionTreeDemo;
