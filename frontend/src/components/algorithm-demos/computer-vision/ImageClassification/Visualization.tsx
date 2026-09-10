import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface Prediction {
  class_name: string;
  class_id: number;
  confidence: number;
  probability: number;
}

interface TrainingResult {
  success: boolean;
  predictions: Prediction[];
  statistics: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const chartData = result.predictions.map((pred) => ({
    class: pred.class_name,
    confidence: Math.round(pred.confidence * 100),
  }));

  return (
    <div className="space-y-4">
      {/* Predictions Bar Chart */}
      <div className="h-96 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 20, bottom: 60, left: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
            <XAxis
              dataKey="class"
              angle={-45}
              textAnchor="end"
              height={100}
              interval={0}
              className="text-gray-700 dark:text-gray-300 text-xs"
            />
            <YAxis
              label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft' }}
              domain={[0, 100]}
              className="text-gray-700 dark:text-gray-300"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#f3f4f6',
                border: '1px solid #d1d5db',
                borderRadius: '0.375rem',
              }}
              formatter={(value) => `${value}%`}
            />
            <Legend />
            <Bar
              dataKey="confidence"
              fill="#3b82f6"
              name="Confidence"
              fillOpacity={0.8}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Model Info
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <strong>Model:</strong> {result.parameters_used.model_name}
            </p>
            <p>
              <strong>Parameters:</strong>{' '}
              {result.model_info.total_parameters
                ? (result.model_info.total_parameters / 1e6).toFixed(1) + 'M'
                : 'N/A'}
            </p>
            <p>
              <strong>Inference Time:</strong> {result.execution_time_ms.toFixed(2)} ms
            </p>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Statistics
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <strong>Top Confidence:</strong>{' '}
              {(result.statistics.top_confidence * 100).toFixed(1)}%
            </p>
            <p>
              <strong>Spread:</strong>{' '}
              {(result.statistics.confidence_spread * 100).toFixed(1)}%
            </p>
            <p>
              <strong>Entropy:</strong> {result.statistics.entropy.toFixed(3)}
            </p>
          </div>
        </div>
      </div>

      {/* Prediction Details */}
      <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          All Predictions
        </h4>
        <div className="space-y-2">
          {result.predictions.map((pred, idx) => (
            <div key={idx} className="flex items-center justify-between text-sm">
              <span className="text-gray-700 dark:text-gray-300">{pred.class_name}</span>
              <div className="flex items-center gap-2">
                <div className="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full"
                    style={{ width: `${pred.confidence * 100}%` }}
                  />
                </div>
                <span className="text-xs text-gray-600 dark:text-gray-400 w-12 text-right">
                  {(pred.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> The model predicts the image contains the classes
          listed above, ranked by confidence. The top prediction is usually the most reliable,
          but lower predictions can be useful context.
        </p>
      </div>
    </div>
  );
}
