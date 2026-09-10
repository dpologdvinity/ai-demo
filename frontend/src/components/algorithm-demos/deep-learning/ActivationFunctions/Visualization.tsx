import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  function_data: Record<string, { x: number[]; y: number[]; derivative: number[] }>;
  comparison_table: Array<Record<string, any>>;
  dead_neuron_demo: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

const FUNCTION_COLORS: Record<string, string> = {
  relu: '#ef4444',
  leaky_relu: '#f97316',
  sigmoid: '#3b82f6',
  tanh: '#8b5cf6',
  elu: '#10b981',
  swish: '#ec4899',
};

export function Visualization({ result }: VisualizationProps) {
  const chartData = useMemo(() => {
    if (!result.function_data || Object.keys(result.function_data).length === 0) {
      return [];
    }

    const firstFunc = Object.values(result.function_data)[0];
    const xValues = firstFunc.x;

    return xValues.map((x, idx) => {
      const point: Record<string, any> = { x: parseFloat(x.toFixed(3)) };
      Object.entries(result.function_data).forEach(([name, data]) => {
        point[name] = parseFloat(data.y[idx].toFixed(4));
      });
      return point;
    });
  }, [result.function_data]);

  const derivativeData = useMemo(() => {
    if (!result.function_data || Object.keys(result.function_data).length === 0) {
      return [];
    }

    const firstFunc = Object.values(result.function_data)[0];
    const xValues = firstFunc.x;

    return xValues.map((x, idx) => {
      const point: Record<string, any> = { x: parseFloat(x.toFixed(3)) };
      Object.entries(result.function_data).forEach(([name, data]) => {
        point[name] = parseFloat(data.derivative[idx].toFixed(4));
      });
      return point;
    });
  }, [result.function_data]);

  const functionNames = Object.keys(result.function_data);

  return (
    <div className="space-y-6">
      {/* Activation Functions */}
      <div className="space-y-4">
        <h3 className="font-semibold text-gray-900 dark:text-gray-100">Activation Functions</h3>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
              <XAxis
                dataKey="x"
                label={{ value: 'Input', position: 'insideBottomRight', offset: -5 }}
                className="text-gray-700 dark:text-gray-300"
              />
              <YAxis
                label={{ value: 'Output', angle: -90, position: 'insideLeft' }}
                className="text-gray-700 dark:text-gray-300"
              />
              <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #4b5563' }} />
              <Legend />
              {functionNames.map((name) => (
                <Line
                  key={name}
                  type="monotone"
                  dataKey={name}
                  stroke={FUNCTION_COLORS[name as keyof typeof FUNCTION_COLORS] || '#6b7280'}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Derivatives */}
      <div className="space-y-4">
        <h3 className="font-semibold text-gray-900 dark:text-gray-100">Derivatives (Gradients)</h3>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={derivativeData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
              <XAxis
                dataKey="x"
                label={{ value: 'Input', position: 'insideBottomRight', offset: -5 }}
                className="text-gray-700 dark:text-gray-300"
              />
              <YAxis
                label={{ value: 'Gradient', angle: -90, position: 'insideLeft' }}
                className="text-gray-700 dark:text-gray-300"
              />
              <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #4b5563' }} />
              <Legend />
              {functionNames.map((name) => (
                <Line
                  key={name}
                  type="monotone"
                  dataKey={name}
                  stroke={FUNCTION_COLORS[name as keyof typeof FUNCTION_COLORS] || '#6b7280'}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Properties Comparison */}
      {result.comparison_table && result.comparison_table.length > 0 && (
        <div className="space-y-4">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100">Properties Comparison</h3>
          <Card>
            <CardContent className="pt-6">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-gray-200 dark:border-gray-700">
                    <tr>
                      {result.comparison_table[0] &&
                        Object.keys(result.comparison_table[0]).map((key) => (
                          <th key={key} className="text-left py-2 px-4 text-gray-700 dark:text-gray-300 font-semibold">
                            {key}
                          </th>
                        ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.comparison_table.map((row, idx) => (
                      <tr key={idx} className="border-b border-gray-200 dark:border-gray-700">
                        {Object.values(row).map((val, i) => (
                          <td key={i} className="py-2 px-4 text-gray-600 dark:text-gray-400">
                            {typeof val === 'boolean' ? (val ? '✓' : '✗') : String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Dead Neuron Demo */}
      {result.dead_neuron_demo && Object.keys(result.dead_neuron_demo).length > 0 && (
        <Card className="bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800">
          <CardHeader>
            <CardTitle className="text-yellow-900 dark:text-yellow-100">Dead Neuron Problem (ReLU)</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-yellow-800 dark:text-yellow-200">
            {Object.entries(result.dead_neuron_demo).map(([key, val]) => (
              <p key={key}>
                <span className="font-medium">{key}:</span> {String(val)}
              </p>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
