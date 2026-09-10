import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface LossHistory {
  iteration: number;
  total_loss: number;
  content_loss: number;
  style_loss: number;
}

interface TrainingResult {
  success: boolean;
  statistics: Record<string, any>;
  visualization_data: Record<string, any>;
  loss_history: LossHistory[];
  feature_visualizations?: any[];
  execution_time_ms: number;
  model_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  // Prepare data for loss curve - sample every n-th point for readability
  const lossData = (result.loss_history || [])
    .filter((_, idx) => idx % Math.max(1, Math.floor(result.loss_history.length / 20)) === 0)
    .map((item) => ({
      iteration: item.iteration,
      total: item.total_loss,
      content: item.content_loss,
      style: item.style_loss,
    }));

  return (
    <div className="space-y-4">
      {/* Loss Curve */}
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={lossData}
            margin={{ top: 20, right: 20, bottom: 60, left: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
            <XAxis
              dataKey="iteration"
              label={{ value: 'Iteration', position: 'insideBottomRight', offset: -10 }}
              className="text-gray-700 dark:text-gray-300"
            />
            <YAxis
              scale="log"
              label={{ value: 'Loss (log scale)', angle: -90, position: 'insideLeft' }}
              className="text-gray-700 dark:text-gray-300"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#f3f4f6',
                border: '1px solid #d1d5db',
                borderRadius: '0.375rem',
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="total"
              stroke="#ef4444"
              name="Total Loss"
              dot={false}
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="content"
              stroke="#3b82f6"
              name="Content Loss"
              dot={false}
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="style"
              stroke="#f59e0b"
              name="Style Loss"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Statistics Grid */}
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Loss Summary
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <strong>Final Total Loss:</strong> {result.statistics.final_total_loss.toFixed(4)}
            </p>
            <p>
              <strong>Content Loss:</strong> {result.statistics.final_content_loss.toFixed(4)}
            </p>
            <p>
              <strong>Style Loss:</strong> {result.statistics.final_style_loss.toFixed(4)}
            </p>
            <p>
              <strong>Loss Reduction:</strong> {result.statistics.loss_reduction.toFixed(1)}%
            </p>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Optimization Details
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <strong>Total Iterations:</strong> {result.statistics.total_iterations}
            </p>
            <p>
              <strong>Convergence Rate:</strong>{' '}
              {(result.statistics.convergence_rate * 100).toFixed(2)}%
            </p>
            <p>
              <strong>Execution Time:</strong> {result.execution_time_ms.toFixed(2)} ms
            </p>
            <p>
              <strong>Image Size:</strong> {result.parameters_used.image_size}x
              {result.parameters_used.image_size}
            </p>
          </div>
        </div>
      </div>

      {/* Parameters Used */}
      <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Style Transfer Parameters
        </h4>
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <p className="text-gray-600 dark:text-gray-400">
              <strong>Content Weight:</strong> {result.parameters_used.content_weight}
            </p>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              <strong>Style Weight:</strong> {result.parameters_used.style_weight.toExponential(2)}
            </p>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              <strong>Learning Rate:</strong> {result.parameters_used.learning_rate}
            </p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">
              <strong>Content Image:</strong> {result.parameters_used.content_image_index}
            </p>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              <strong>Style Image:</strong> {result.parameters_used.style_image_index}
            </p>
          </div>
        </div>
      </div>

      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> The loss curve shows how well the network is learning
          to balance content and style. Decreasing loss indicates convergence. The log scale helps
          visualize both large and small loss values. Lower final loss generally means better
          stylization.
        </p>
      </div>
    </div>
  );
}
