import React, { useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ZAxis,
} from 'recharts';

interface TrainingResult {
  success: boolean;
  embedded_data: number[][];
  labels: number[];
  kl_divergence: number;
  visualization_data: {
    type: '2d' | '3d';
    scatter_data: Array<{
      x: number;
      y: number;
      z?: number;
      label: number;
    }>;
    x_label: string;
    y_label: string;
    z_label?: string;
  };
  execution_time_ms: number;
  model_info: {
    n_components: number;
    perplexity: number;
    learning_rate: number;
    n_iter: number;
    n_iter_final: number;
    kl_divergence: number;
  };
}

interface VisualizationProps {
  result: TrainingResult;
}

// Color palette for different digit classes (0-9)
const DIGIT_COLORS = [
  '#e41a1c', // 0 - Red
  '#377eb8', // 1 - Blue
  '#4daf4a', // 2 - Green
  '#984ea3', // 3 - Purple
  '#ff7f00', // 4 - Orange
  '#ffff33', // 5 - Yellow
  '#a65628', // 6 - Brown
  '#f781bf', // 7 - Pink
  '#999999', // 8 - Gray
  '#66c2a5', // 9 - Teal
];

export function Visualization({ result }: VisualizationProps) {
  const { visualization_data, model_info } = result;

  // Group data by label for colored visualization
  const groupedData = useMemo(() => {
    const groups: { [key: number]: any[] } = {};

    visualization_data.scatter_data.forEach((point) => {
      const label = point.label;
      if (!groups[label]) {
        groups[label] = [];
      }
      groups[label].push(point);
    });

    return groups;
  }, [visualization_data.scatter_data]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
          <p className="text-sm font-semibold mb-1">Digit: {data.label}</p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            {visualization_data.x_label}: {data.x.toFixed(3)}
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            {visualization_data.y_label}: {data.y.toFixed(3)}
          </p>
          {data.z !== undefined && (
            <p className="text-xs text-gray-600 dark:text-gray-400">
              {visualization_data.z_label}: {data.z.toFixed(3)}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  if (visualization_data.type === '2d') {
    return (
      <div className="space-y-4">
        <div className="h-[500px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart
              margin={{ top: 20, right: 20, bottom: 60, left: 60 }}
            >
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
              <XAxis
                type="number"
                dataKey="x"
                name={visualization_data.x_label}
                label={{
                  value: visualization_data.x_label,
                  position: 'bottom',
                  offset: 40,
                }}
                className="text-gray-700 dark:text-gray-300"
              />
              <YAxis
                type="number"
                dataKey="y"
                name={visualization_data.y_label}
                label={{
                  value: visualization_data.y_label,
                  angle: -90,
                  position: 'left',
                  offset: 40,
                }}
                className="text-gray-700 dark:text-gray-300"
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ paddingTop: '20px' }}
                payload={Object.keys(groupedData).map((label) => ({
                  value: `Digit ${label}`,
                  type: 'circle',
                  color: DIGIT_COLORS[parseInt(label)],
                }))}
              />
              {Object.entries(groupedData).map(([label, data]) => (
                <Scatter
                  key={label}
                  name={`Digit ${label}`}
                  data={data}
                  fill={DIGIT_COLORS[parseInt(label)]}
                  fillOpacity={0.6}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="grid grid-cols-2 gap-4 mt-4">
          <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Model Information
            </h4>
            <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
              <p>Perplexity: {model_info.perplexity}</p>
              <p>Learning Rate: {model_info.learning_rate}</p>
              <p>Iterations: {model_info.n_iter_final}</p>
            </div>
          </div>

          <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Quality Metrics
            </h4>
            <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
              <p>KL Divergence: {model_info.kl_divergence.toFixed(4)}</p>
              <p>Data Points: {result.labels.length}</p>
              <p>Classes: 10 (digits 0-9)</p>
            </div>
          </div>
        </div>

        <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <p className="text-sm text-blue-900 dark:text-blue-100">
            <strong>Interpretation:</strong> Points closer together represent digits with similar
            visual features. Well-separated clusters indicate distinct digit classes. Lower KL
            divergence indicates better preservation of high-dimensional relationships.
          </p>
        </div>
      </div>
    );
  }

  // For 3D visualization, show a message (3D scatter would require a different library)
  return (
    <div className="space-y-4">
      <div className="p-8 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg text-center">
        <h4 className="text-lg font-semibold text-yellow-900 dark:text-yellow-100 mb-2">
          3D Visualization
        </h4>
        <p className="text-sm text-yellow-800 dark:text-yellow-200 mb-4">
          3D scatter plot visualization is coming soon. For now, view the 2D projection or check
          the data table below.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Model Information
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Dimensions: 3D</p>
            <p>Perplexity: {model_info.perplexity}</p>
            <p>Learning Rate: {model_info.learning_rate}</p>
            <p>Iterations: {model_info.n_iter_final}</p>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Quality Metrics
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>KL Divergence: {model_info.kl_divergence.toFixed(4)}</p>
            <p>Data Points: {result.labels.length}</p>
            <p>Classes: 10 (digits 0-9)</p>
          </div>
        </div>
      </div>
    </div>
  );
}
