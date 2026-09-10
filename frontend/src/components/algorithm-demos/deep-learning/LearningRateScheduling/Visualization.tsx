import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  schedule_data: Record<string, number[]>;
  loss_data: Record<string, number[]>;
  convergence_metrics: Record<string, Record<string, any>>;
  comparison_table: Array<Record<string, any>>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

const SCHEDULE_COLORS: Record<string, string> = {
  step: '#3b82f6',
  exponential: '#ef4444',
  cosine: '#8b5cf6',
  reduce_on_plateau: '#f59e0b',
  cyclic: '#10b981',
};

export function Visualization({ result }: VisualizationProps) {
  const scheduleChartData = useMemo(() => {
    if (!result.schedule_data || Object.keys(result.schedule_data).length === 0) {
      return [];
    }

    const schedules = result.schedule_data;
    const maxLen = Math.max(...Object.values(schedules).map((arr) => arr.length));

    return Array.from({ length: maxLen }, (_, i) => {
      const point: Record<string, any> = { epoch: i };
      Object.entries(schedules).forEach(([name, lrs]) => {
        if (i < lrs.length) {
          point[name] = parseFloat(lrs[i].toFixed(6));
        }
      });
      return point;
    });
  }, [result.schedule_data]);

  const lossChartData = useMemo(() => {
    if (!result.loss_data || Object.keys(result.loss_data).length === 0) {
      return [];
    }

    const losses = result.loss_data;
    const maxLen = Math.max(...Object.values(losses).map((arr) => arr.length));

    return Array.from({ length: maxLen }, (_, i) => {
      const point: Record<string, any> = { epoch: i };
      Object.entries(losses).forEach(([name, vals]) => {
        if (i < vals.length) {
          point[name] = parseFloat(vals[i].toFixed(4));
        }
      });
      return point;
    });
  }, [result.loss_data]);

  const scheduleNames = Object.keys(result.schedule_data || {});
  const lossNames = Object.keys(result.loss_data || {});

  return (
    <div className="space-y-6">
      {/* Learning Rate Schedules */}
      {scheduleChartData.length > 0 && (
        <div className="space-y-4">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100">Learning Rate Schedules</h3>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={scheduleChartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                <XAxis
                  dataKey="epoch"
                  label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5 }}
                  className="text-gray-700 dark:text-gray-300"
                />
                <YAxis
                  scale="log"
                  label={{ value: 'Learning Rate (log)', angle: -90, position: 'insideLeft' }}
                  className="text-gray-700 dark:text-gray-300"
                />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #4b5563' }} />
                <Legend />
                {scheduleNames.map((name) => (
                  <Line
                    key={name}
                    type="monotone"
                    dataKey={name}
                    stroke={SCHEDULE_COLORS[name as keyof typeof SCHEDULE_COLORS] || '#6b7280'}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Training Loss */}
      {lossChartData.length > 0 && (
        <div className="space-y-4">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100">Training Loss</h3>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lossChartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                <XAxis
                  dataKey="epoch"
                  label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5 }}
                  className="text-gray-700 dark:text-gray-300"
                />
                <YAxis
                  label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                  className="text-gray-700 dark:text-gray-300"
                />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #4b5563' }} />
                <Legend />
                {lossNames.map((name) => (
                  <Line
                    key={name}
                    type="monotone"
                    dataKey={name}
                    stroke={SCHEDULE_COLORS[name as keyof typeof SCHEDULE_COLORS] || '#6b7280'}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Comparison Table */}
      {result.comparison_table && result.comparison_table.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Schedule Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-gray-200 dark:border-gray-700">
                  <tr>
                    <th className="text-left py-2 px-4 text-gray-700 dark:text-gray-300 font-semibold">
                      Schedule
                    </th>
                    <th className="text-left py-2 px-4 text-gray-700 dark:text-gray-300 font-semibold">
                      Final Loss
                    </th>
                    <th className="text-left py-2 px-4 text-gray-700 dark:text-gray-300 font-semibold">
                      Convergence Epoch
                    </th>
                    <th className="text-left py-2 px-4 text-gray-700 dark:text-gray-300 font-semibold">
                      Training Time (ms)
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.comparison_table.map((row, idx) => (
                    <tr key={idx} className="border-b border-gray-200 dark:border-gray-700">
                      <td className="py-2 px-4 text-gray-600 dark:text-gray-400 font-medium">
                        {row.schedule}
                      </td>
                      <td className="py-2 px-4 text-gray-600 dark:text-gray-400">
                        {row.final_loss?.toFixed(4)}
                      </td>
                      <td className="py-2 px-4 text-gray-600 dark:text-gray-400">
                        {row.convergence_epoch || 'N/A'}
                      </td>
                      <td className="py-2 px-4 text-gray-600 dark:text-gray-400">
                        {row.training_time?.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
