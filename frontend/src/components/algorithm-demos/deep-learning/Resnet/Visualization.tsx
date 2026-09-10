import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface PredictionResult {
  class_id: number;
  class_name: string;
  confidence: number;
}

interface TrainingResult {
  success: boolean;
  predictions: PredictionResult[];
  input_image?: number[][][];
  input_shape: number[];
  feature_maps?: Record<string, unknown>;
  visualization_data: Record<string, unknown>;
  execution_time_ms: number;
  model_info: Record<string, unknown>;
  parameters_used: Record<string, unknown>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { predictions, model_info, execution_time_ms } = result;

  const chartData = predictions.map((pred) => ({
    name: pred.class_name,
    confidence: Math.round(pred.confidence * 10000) / 100,
    class_id: pred.class_id,
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
          <p className="text-sm font-semibold mb-1">{data.name}</p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            Confidence: {data.confidence}%
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            Class ID: {data.class_id}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-4">
      <div className="h-[400px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 20, bottom: 60, left: 60 }}
          >
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
            <XAxis
              dataKey="name"
              angle={-45}
              textAnchor="end"
              height={100}
              className="text-gray-700 dark:text-gray-300"
            />
            <YAxis
              label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft' }}
              className="text-gray-700 dark:text-gray-300"
              domain={[0, 100]}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Bar
              dataKey="confidence"
              fill="#3b82f6"
              name="Confidence Score"
              radius={[8, 8, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Model Information
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Model: {(model_info?.model_variant as string) || 'resnet18'}</p>
            <p>
              Parameters: {new Intl.NumberFormat().format(
                (model_info?.total_parameters as number) || 0
              )}
            </p>
            <p>Depth: {(model_info?.depth as number) || 18}</p>
            <p>
              Residual Blocks: {(model_info?.num_residual_blocks as number) || 8}
            </p>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Performance
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Execution Time: {execution_time_ms.toFixed(2)} ms</p>
            <p>
              Top Prediction: {predictions[0]?.class_name}
            </p>
            <p>
              Confidence: {(predictions[0]?.confidence * 100).toFixed(2)}%
            </p>
            <p>Predictions Returned: {predictions.length}</p>
          </div>
        </div>
      </div>

      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> ResNet uses residual connections (skip connections) that allow
          gradients to flow directly through deep networks. This architecture achieves state-of-the-art
          performance on image classification tasks while being computationally efficient.
        </p>
      </div>
    </div>
  );
}
