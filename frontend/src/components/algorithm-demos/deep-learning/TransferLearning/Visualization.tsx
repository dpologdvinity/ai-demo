import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface LayerInfo {
  name: string;
  trainable: boolean;
  num_params: number;
  frozen: boolean;
}

interface TrainingResult {
  success: boolean;
  metrics: Record<string, number>;
  training_history: Record<string, number[]>;
  confusion_matrix: number[][];
  layer_info: LayerInfo[];
  strategy_comparison?: Record<string, unknown>[];
  feature_maps?: Record<string, unknown>;
  sample_predictions: Record<string, unknown>[];
  visualization_data: Record<string, unknown>;
  execution_time_ms: number;
  model_info: Record<string, unknown>;
  parameters_used: Record<string, unknown>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { metrics, training_history, model_info, layer_info } = result;

  const chartData = useMemo(() => {
    const epochs = (training_history?.epochs as number[]) || [];
    const trainLoss = (training_history?.train_loss as number[]) || [];
    const valLoss = (training_history?.val_loss as number[]) || [];
    const trainAcc = (training_history?.train_accuracy as number[]) || [];
    const valAcc = (training_history?.val_accuracy as number[]) || [];

    return epochs.map((epoch, idx) => ({
      epoch,
      train_loss: trainLoss[idx] || 0,
      val_loss: valLoss[idx] || 0,
      train_accuracy: trainAcc[idx] || 0,
      val_accuracy: valAcc[idx] || 0,
    }));
  }, [training_history]);

  const frozenLayers = layer_info?.filter((l) => l.frozen).length || 0;
  const trainableLayers = layer_info?.filter((l) => l.trainable).length || 0;
  const frozenParams = layer_info?.reduce((sum, l) => sum + (l.frozen ? l.num_params : 0), 0) || 0;
  const trainableParams = layer_info?.reduce((sum, l) => sum + (l.trainable ? l.num_params : 0), 0) || 0;

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-xs text-gray-600 dark:text-gray-400">
              {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {chartData.length > 0 && (
        <>
          <div>
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Training & Validation Loss
            </h4>
            <div className="h-[350px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={chartData}
                  margin={{ top: 10, right: 20, bottom: 30, left: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                  <XAxis
                    dataKey="epoch"
                    label={{ value: 'Epoch', position: 'insideBottomRight', offset: -10 }}
                    className="text-gray-700 dark:text-gray-300"
                  />
                  <YAxis
                    label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                    className="text-gray-700 dark:text-gray-300"
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="train_loss"
                    stroke="#ef4444"
                    strokeWidth={2}
                    name="Training Loss"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="val_loss"
                    stroke="#f97316"
                    strokeWidth={2}
                    name="Validation Loss"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Training & Validation Accuracy
            </h4>
            <div className="h-[350px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={chartData}
                  margin={{ top: 10, right: 20, bottom: 30, left: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                  <XAxis
                    dataKey="epoch"
                    label={{ value: 'Epoch', position: 'insideBottomRight', offset: -10 }}
                    className="text-gray-700 dark:text-gray-300"
                  />
                  <YAxis
                    label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }}
                    className="text-gray-700 dark:text-gray-300"
                    domain={[0, 1]}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="train_accuracy"
                    stroke="#22c55e"
                    strokeWidth={2}
                    name="Training Accuracy"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="val_accuracy"
                    stroke="#16a34a"
                    strokeWidth={2}
                    name="Validation Accuracy"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Model Configuration
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Base Model: {(model_info?.base_model as string) || 'resnet18'}</p>
            <p>
              Total Params: {new Intl.NumberFormat().format(
                (model_info?.total_parameters as number) || 0
              )}
            </p>
            <p>
              Trainable: {new Intl.NumberFormat().format(trainableParams)}
            </p>
            <p>
              Frozen: {new Intl.NumberFormat().format(frozenParams)}
            </p>
            <p>Classes: {(model_info?.num_classes as number) || 'N/A'}</p>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Final Metrics
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              Train Accuracy: {(metrics?.train_accuracy as number * 100)?.toFixed(2) || 'N/A'}%
            </p>
            <p>
              Val Accuracy: {(metrics?.val_accuracy as number * 100)?.toFixed(2) || 'N/A'}%
            </p>
            <p>
              Test Accuracy: {(metrics?.test_accuracy as number * 100)?.toFixed(2) || 'N/A'}%
            </p>
            <p>
              Final Loss: {(metrics?.val_loss as number)?.toFixed(4) || 'N/A'}
            </p>
          </div>
        </div>
      </div>

      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
          Layer Freezing Summary
        </h4>
        <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
          <p>Frozen Layers: {frozenLayers} | Trainable Layers: {trainableLayers}</p>
          <p>
            Frozen Parameters: {new Intl.NumberFormat().format(frozenParams)} |
            Trainable: {new Intl.NumberFormat().format(trainableParams)}
          </p>
        </div>
      </div>

      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> Training curves show how loss decreases and accuracy
          improves over epochs. Well-trained models show decreasing training and validation loss
          with increasing accuracy. The gap between training and validation metrics indicates
          generalization: small gaps suggest good generalization, large gaps suggest overfitting.
        </p>
      </div>
    </div>
  );
}
