import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  metrics: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
  parameters_used: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

const OPTIMIZER_COLORS: Record<string, string> = {
  Adam: '#3b82f6',
  SGD: '#ef4444',
  Momentum: '#f59e0b',
  RMSprop: '#10b981',
};

export function Visualization({ result }: VisualizationProps) {
  const convergenceData = useMemo(() => {
    if (
      !result.visualization_data.convergence_comparison ||
      Object.keys(result.visualization_data.convergence_comparison).length === 0
    ) {
      return [];
    }

    const comparisonData = result.visualization_data.convergence_comparison;
    const maxLen = Math.max(...Object.values(comparisonData).map((arr: any) => (Array.isArray(arr) ? arr.length : 0)));

    return Array.from({ length: maxLen }, (_, i) => {
      const point: Record<string, any> = { iteration: i };
      Object.entries(comparisonData).forEach(([name, losses]: [string, any]) => {
        if (i < losses.length) {
          point[name] = parseFloat(losses[i].toFixed(6));
        }
      });
      return point;
    });
  }, [result.visualization_data.convergence_comparison]);

  const hasMetrics = result.metrics && Object.keys(result.metrics).length > 0;

  return (
    <div className="space-y-6">
      {/* Convergence Curves */}
      {convergenceData.length > 0 && (
        <div className="space-y-4">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100">Convergence Comparison</h3>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={convergenceData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                <XAxis
                  dataKey="iteration"
                  label={{ value: 'Iteration', position: 'insideBottomRight', offset: -5 }}
                  className="text-gray-700 dark:text-gray-300"
                />
                <YAxis
                  scale="log"
                  label={{ value: 'Loss (log scale)', angle: -90, position: 'insideLeft' }}
                  className="text-gray-700 dark:text-gray-300"
                />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #4b5563' }} />
                <Legend />
                {Object.keys(result.visualization_data.convergence_comparison || {}).map((name) => (
                  <Line
                    key={name}
                    type="monotone"
                    dataKey={name}
                    stroke={OPTIMIZER_COLORS[name as keyof typeof OPTIMIZER_COLORS] || '#6b7280'}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Performance Metrics Table */}
      {hasMetrics && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-gray-200 dark:border-gray-700">
                  <tr>
                    <th className="text-left py-2 px-4 text-gray-700 dark:text-gray-300 font-semibold">
                      Optimizer
                    </th>
                    <th className="text-left py-2 px-4 text-gray-700 dark:text-gray-300 font-semibold">
                      Final Loss
                    </th>
                    <th className="text-left py-2 px-4 text-gray-700 dark:text-gray-300 font-semibold">
                      Iterations
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { name: 'Adam', finalKey: 'adam_final_loss', iterKey: 'adam_iterations' },
                    { name: 'SGD', finalKey: 'sgd_final_loss', iterKey: 'sgd_iterations' },
                    { name: 'Momentum', finalKey: 'momentum_final_loss', iterKey: 'momentum_iterations' },
                    { name: 'RMSprop', finalKey: 'rmsprop_final_loss', iterKey: 'rmsprop_iterations' },
                  ].map((opt) => {
                    const finalLoss = result.metrics[opt.finalKey];
                    const iters = result.metrics[opt.iterKey];
                    if (finalLoss === undefined && iters === undefined) return null;
                    return (
                      <tr key={opt.name} className="border-b border-gray-200 dark:border-gray-700">
                        <td className="py-2 px-4 text-gray-600 dark:text-gray-400 font-medium">
                          {opt.name}
                        </td>
                        <td className="py-2 px-4 text-gray-600 dark:text-gray-400">
                          {finalLoss !== undefined ? finalLoss.toFixed(6) : 'N/A'}
                        </td>
                        <td className="py-2 px-4 text-gray-600 dark:text-gray-400">
                          {iters !== undefined ? iters : 'N/A'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Model Info */}
      {result.model_info && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Optimization Function</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {result.model_info.function_name && (
              <p>
                <span className="font-medium">Name:</span> {result.model_info.function_name}
              </p>
            )}
            {result.model_info.function_description && (
              <p>
                <span className="font-medium">Description:</span> {result.model_info.function_description}
              </p>
            )}
            {result.model_info.optimal_value && (
              <p>
                <span className="font-medium">Optimal Value:</span> {result.model_info.optimal_value}
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
