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
  const { training_history, predictions, actual, visualization_data, metrics } = result;

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

  // Visualize attention weights if available
  const attentionData = useMemo(() => {
    if (!visualization_data?.attention_heatmap?.weights || visualization_data.attention_heatmap.weights.length === 0) {
      return null;
    }
    return visualization_data.attention_heatmap;
  }, [visualization_data]);

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
          yLabel="Loss"
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
          title="Predictions vs Actual Sequences"
          xLabel="Position"
          yLabel="Token ID"
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
              <p className="text-sm text-gray-600 dark:text-gray-400">Accuracy</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {(metrics.accuracy * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">Test Loss</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {metrics.test_loss?.toFixed(4) ?? 'N/A'}
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">Perplexity</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {metrics.perplexity?.toFixed(2) ?? 'N/A'}
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">Execution Time</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {(result.execution_time_ms / 1000).toFixed(2)} s
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Attention Heatmap */}
      {attentionData && (
        <Card>
          <CardHeader>
            <CardTitle>Attention Weights Visualization</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto text-xs">
              <p className="text-gray-600 dark:text-gray-400 mb-3">
                Multi-head attention weights showing which input positions the model focuses on
              </p>
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <p className="text-gray-700 dark:text-gray-300">
                  Attention weights shape: {attentionData.weights.length} positions x {attentionData.weights[0]?.length ?? 0} attention connections
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Model Info */}
      <Card>
        <CardHeader>
          <CardTitle>Model Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Model Dimension:</span>
              <span className="font-medium">{result.model_info?.d_model ?? 'N/A'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Attention Heads:</span>
              <span className="font-medium">{result.model_info?.nhead ?? 'N/A'}</span>
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
