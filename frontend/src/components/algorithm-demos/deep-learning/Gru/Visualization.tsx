import React, { useMemo } from 'react';
import { LineChart } from '@/components/visualizations/LineChart';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  metrics: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { metrics, visualization_data } = result;

  // Format training curves from visualization_data
  const trainingData = useMemo(() => {
    if (!visualization_data?.training_curves) {
      return [];
    }
    const curves = visualization_data.training_curves;
    const keys = Object.keys(curves);
    if (keys.length === 0) return [];

    const firstKey = keys[0];
    const length = curves[firstKey].length;

    return Array.from({ length }, (_, idx) => ({
      epoch: idx + 1,
      ...Object.fromEntries(
        keys.map(key => [key, curves[key][idx] ?? 0])
      ),
    }));
  }, [visualization_data]);

  // Format predictions from visualization_data
  const predictionsData = useMemo(() => {
    if (!visualization_data?.predictions || visualization_data.predictions.length === 0) {
      return [];
    }
    return visualization_data.predictions.slice(0, 50); // Show first 50 predictions
  }, [visualization_data]);

  // Determine which keys to display for training curves
  const trainingCurveKeys = useMemo(() => {
    if (trainingData.length === 0) return [];
    const firstEntry = trainingData[0];
    return Object.keys(firstEntry).filter(key => key !== 'epoch');
  }, [trainingData]);

  return (
    <div className="space-y-6">
      {/* Training Curves */}
      {trainingData.length > 0 && trainingCurveKeys.length > 0 && (
        <LineChart
          data={trainingData}
          xKey="epoch"
          yKey={trainingCurveKeys}
          title="Training Curves"
          xLabel="Epoch"
          yLabel="Loss / Metrics"
          height={400}
        />
      )}

      {/* Predictions */}
      {predictionsData.length > 0 && (
        <LineChart
          data={predictionsData}
          xKey="step"
          yKey={['predicted', 'actual']}
          title="Predictions vs Actual"
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
              <p className="text-sm text-gray-600 dark:text-gray-400">Train Loss</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {metrics.train_loss?.toFixed(4) ?? 'N/A'}
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">Test Loss</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {metrics.test_loss?.toFixed(4) ?? 'N/A'}
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">Test MSE</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {metrics.test_mse?.toFixed(4) ?? 'N/A'}
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">Test MAE</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {metrics.test_mae?.toFixed(4) ?? 'N/A'}
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
              <span className="text-gray-600 dark:text-gray-400">Dropout Rate:</span>
              <span className="font-medium">{result.model_info?.dropout ?? 'N/A'}</span>
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
