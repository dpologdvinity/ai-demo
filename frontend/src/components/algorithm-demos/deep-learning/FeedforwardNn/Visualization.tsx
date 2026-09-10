import React, { useMemo } from 'react';
import { LineChart } from '@/components/visualizations/LineChart';
import { ConfusionMatrix } from '@/components/visualizations/ConfusionMatrix';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  metrics: Record<string, any>;
  predictions: number[];
  actual: number[];
  training_history: Record<string, number[]>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { training_history, metrics, visualization_data } = result;

  // Format training history for LineChart
  const trainingData = useMemo(() => {
    if (!training_history || !training_history.iterations) {
      return [];
    }
    return training_history.iterations.map((iter: number, idx: number) => ({
      iteration: iter,
      loss: training_history.loss?.[idx] ?? 0,
      accuracy: training_history.accuracy?.[idx] ?? 0,
    }));
  }, [training_history]);

  // Format confusion matrix from visualization_data
  const confusionMatrixData = useMemo(() => {
    if (visualization_data?.confusion_matrix) {
      return visualization_data.confusion_matrix;
    }
    return null;
  }, [visualization_data]);

  return (
    <div className="space-y-6">
      {/* Training Curves */}
      {trainingData.length > 0 && (
        <div>
          <LineChart
            data={trainingData}
            xKey="iteration"
            yKey={['loss', 'accuracy']}
            title="Training Curves"
            xLabel="Iteration"
            yLabel="Value"
            height={400}
            colors={['#ef4444', '#3b82f6']}
          />
        </div>
      )}

      {/* Confusion Matrix */}
      {confusionMatrixData && (
        <ConfusionMatrix
          matrix={confusionMatrixData}
          labels={['Class 0', 'Class 1', 'Class 2']}
          title="Confusion Matrix"
        />
      )}

      {/* Metrics Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">Accuracy</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {(metrics.accuracy * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">Precision</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {(metrics.precision * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">Recall</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {(metrics.recall * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">F1 Score</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {(metrics.f1_score * 100).toFixed(2)}%
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Info */}
      <Card>
        <CardHeader>
          <CardTitle>Model Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Total Parameters:</span>
              <span className="font-medium">{result.model_info?.total_parameters ?? 'N/A'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Execution Time:</span>
              <span className="font-medium">{result.execution_time_ms.toFixed(2)} ms</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
