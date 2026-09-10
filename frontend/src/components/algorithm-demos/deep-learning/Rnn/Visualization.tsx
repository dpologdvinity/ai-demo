import React, { useMemo } from 'react';
import { LineChart } from '@/components/visualizations/LineChart';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  metrics: Record<string, any>;
  training_history: number[];
  predictions: number[][];
  actual: number[][];
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { training_history, predictions, actual } = result;

  // Format training history for LineChart
  const trainingData = useMemo(() => {
    if (!training_history || training_history.length === 0) {
      return [];
    }
    return training_history.map((loss, idx) => ({
      epoch: idx + 1,
      loss: loss,
    }));
  }, [training_history]);

  // Format predictions vs actual for visualization
  const predictionsData = useMemo(() => {
    if (!predictions || !actual || predictions.length === 0) {
      return [];
    }
    // Take first sequence
    const pred = predictions[0] || [];
    const act = actual[0] || [];
    const maxLen = Math.max(pred.length, act.length);
    return Array.from({ length: maxLen }, (_, idx) => ({
      step: idx + 1,
      predicted: pred[idx] ?? 0,
      actual: act[idx] ?? 0,
    }));
  }, [predictions, actual]);

  return (
    <div className="space-y-6">
      {/* Training Loss */}
      {trainingData.length > 0 && (
        <LineChart
          data={trainingData}
          xKey="epoch"
          yKey="loss"
          title="Training Loss"
          xLabel="Epoch"
          yLabel="Loss (MSE)"
          height={400}
          colors={['#ef4444']}
        />
      )}

      {/* Predictions vs Actual */}
      {predictionsData.length > 0 && (
        <LineChart
          data={predictionsData}
          xKey="step"
          yKey={['predicted', 'actual']}
          title="Predictions vs Actual Values"
          xLabel="Time Step"
          yLabel="Value"
          height={400}
          colors={['#3b82f6', '#10b981']}
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
              <p className="text-sm text-gray-600 dark:text-gray-400">Final Loss</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {result.metrics.final_loss?.toFixed(4) ?? 'N/A'}
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">MSE</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {result.metrics.mse?.toFixed(4) ?? 'N/A'}
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">MAE</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {result.metrics.mae?.toFixed(4) ?? 'N/A'}
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">Execution Time</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {result.execution_time_ms.toFixed(0)} ms
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
              <span className="text-gray-600 dark:text-gray-400">Hidden Size:</span>
              <span className="font-medium">{result.model_info?.hidden_size ?? 'N/A'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Number of Layers:</span>
              <span className="font-medium">{result.model_info?.num_layers ?? 'N/A'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Total Parameters:</span>
              <span className="font-medium">{result.model_info?.total_parameters ?? 'N/A'}</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
