import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
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

interface ClassMetrics {
  class_name: string;
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

interface TrainingResult {
  success: boolean;
  class_names: string[];
  class_metrics: ClassMetrics[];
  overall_metrics: Record<string, number>;
  confusion_matrix: number[][];
  execution_time_ms: number;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  // Prepare class metrics for chart
  const chartData = result.class_metrics.map((metric) => ({
    class: metric.class_name,
    precision: parseFloat((metric.precision * 100).toFixed(2)),
    recall: parseFloat((metric.recall * 100).toFixed(2)),
    f1: parseFloat((metric.f1_score * 100).toFixed(2)),
  }));

  return (
    <div className="space-y-6">
      {/* Overall Metrics */}
      <div className="grid grid-cols-3 gap-3">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400">Accuracy</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {(result.overall_metrics.accuracy * 100).toFixed(1)}%
              </p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400">Classes</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {result.class_names.length}
              </p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400">Execution</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {result.execution_time_ms.toFixed(0)}ms
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Class Metrics Chart */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Per-Class Performance Metrics
        </h3>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
              <XAxis
                dataKey="class"
                angle={-45}
                textAnchor="end"
                height={100}
                className="text-gray-700 dark:text-gray-300"
              />
              <YAxis
                label={{ value: 'Score (%)', angle: -90, position: 'insideLeft' }}
                className="text-gray-700 dark:text-gray-300"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(31, 41, 55, 0.95)',
                  border: '1px solid rgb(75, 85, 99)',
                  borderRadius: '8px',
                  color: 'white',
                }}
                formatter={(value: any) => `${(typeof value === 'number' ? value : 0).toFixed(1)}%`}
              />
              <Legend />
              <Bar dataKey="precision" fill="#3b82f6" name="Precision" radius={[4, 4, 0, 0]} />
              <Bar dataKey="recall" fill="#10b981" name="Recall" radius={[4, 4, 0, 0]} />
              <Bar dataKey="f1" fill="#f59e0b" name="F1-Score" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion Matrix */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Confusion Matrix
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr>
                <th className="border border-gray-300 dark:border-gray-600 p-2 bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
                  Predicted →
                </th>
                {result.class_names.map((name) => (
                  <th
                    key={name}
                    className="border border-gray-300 dark:border-gray-600 p-2 bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 text-center"
                  >
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.class_names.map((actual, i) => (
                <tr key={actual}>
                  <td className="border border-gray-300 dark:border-gray-600 p-2 bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 font-medium">
                    {actual}
                  </td>
                  {result.confusion_matrix[i].map((count, j) => (
                    <td
                      key={j}
                      className="border border-gray-300 dark:border-gray-600 p-2 text-center text-gray-900 dark:text-gray-100"
                      style={{
                        backgroundColor: i === j
                          ? 'rgba(16, 185, 129, 0.1)'
                          : 'rgba(239, 68, 68, 0.1)',
                      }}
                    >
                      {count}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detailed Class Metrics */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Detailed Metrics
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left px-3 py-2 text-gray-700 dark:text-gray-300">Class</th>
                <th className="text-right px-3 py-2 text-gray-700 dark:text-gray-300">Precision</th>
                <th className="text-right px-3 py-2 text-gray-700 dark:text-gray-300">Recall</th>
                <th className="text-right px-3 py-2 text-gray-700 dark:text-gray-300">F1-Score</th>
                <th className="text-right px-3 py-2 text-gray-700 dark:text-gray-300">Support</th>
              </tr>
            </thead>
            <tbody>
              {result.class_metrics.map((metric) => (
                <tr key={metric.class_name} className="border-b border-gray-100 dark:border-gray-800">
                  <td className="px-3 py-2 text-gray-900 dark:text-gray-100 font-medium">
                    {metric.class_name}
                  </td>
                  <td className="text-right px-3 py-2 text-gray-600 dark:text-gray-400">
                    {(metric.precision * 100).toFixed(1)}%
                  </td>
                  <td className="text-right px-3 py-2 text-gray-600 dark:text-gray-400">
                    {(metric.recall * 100).toFixed(1)}%
                  </td>
                  <td className="text-right px-3 py-2 text-gray-600 dark:text-gray-400">
                    {(metric.f1_score * 100).toFixed(1)}%
                  </td>
                  <td className="text-right px-3 py-2 text-gray-600 dark:text-gray-400">
                    {metric.support}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
