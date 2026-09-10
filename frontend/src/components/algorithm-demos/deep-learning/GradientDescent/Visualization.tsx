import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  results?: Array<Record<string, any>>;
  single_result?: Record<string, any>;
  contour_data: Record<string, any>;
  statistics_table: Array<Record<string, any>>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

const OPTIMIZER_COLORS: Record<string, string> = {
  SGD: '#ef4444',
  Momentum: '#f97316',
  RMSprop: '#eab308',
  Adam: '#3b82f6',
  AdaGrad: '#8b5cf6',
};

export function Visualization({ result }: VisualizationProps) {
  const lossChartData = useMemo(() => {
    const data = result.results || (result.single_result ? [result.single_result] : []);
    if (data.length === 0) return [];

    const maxLen = Math.max(...data.map((r) => r.loss_history?.length || 0));
    return Array.from({ length: maxLen }, (_, i) => {
      const point: Record<string, any> = { iteration: i };
      data.forEach((r) => {
        if (i < r.loss_history?.length) {
          point[r.optimizer_name] = parseFloat(r.loss_history[i].toFixed(6));
        }
      });
      return point;
    });
  }, [result.results, result.single_result]);

  return (
    <div className="space-y-6">
      {/* Loss Curves */}
      {lossChartData.length > 0 && (
        <div className="space-y-4">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100">Loss Convergence</h3>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lossChartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
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
                {(result.results || []).map((r) => (
                  <Line
                    key={r.optimizer_name}
                    type="monotone"
                    dataKey={r.optimizer_name}
                    stroke={OPTIMIZER_COLORS[r.optimizer_name as keyof typeof OPTIMIZER_COLORS] || '#6b7280'}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Statistics Table */}
      {result.statistics_table && result.statistics_table.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Optimizer Statistics</CardTitle>
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
                      Convergence Iter
                    </th>
                    <th className="text-left py-2 px-4 text-gray-700 dark:text-gray-300 font-semibold">
                      Path Length
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.statistics_table.map((row, idx) => (
                    <tr key={idx} className="border-b border-gray-200 dark:border-gray-700">
                      <td className="py-2 px-4 text-gray-600 dark:text-gray-400 font-medium">
                        {row.optimizer}
                      </td>
                      <td className="py-2 px-4 text-gray-600 dark:text-gray-400">
                        {row.final_loss?.toFixed(6)}
                      </td>
                      <td className="py-2 px-4 text-gray-600 dark:text-gray-400">
                        {row.iterations_to_converge || 'N/A'}
                      </td>
                      <td className="py-2 px-4 text-gray-600 dark:text-gray-400">
                        {row.path_length?.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Test Function Info */}
      {result.parameters_used && (
        <Card className="bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
          <CardHeader>
            <CardTitle className="text-blue-900 dark:text-blue-100">Test Function</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-blue-800 dark:text-blue-200">
            <p className="capitalize font-medium">{result.parameters_used.test_function}</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
