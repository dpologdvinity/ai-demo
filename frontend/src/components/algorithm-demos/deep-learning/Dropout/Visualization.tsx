import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  metrics: Record<string, any>;
  training_curves: Record<string, any>;
  dropout_comparison: Array<Record<string, any>>;
  overfitting_metrics: Record<string, any>;
  dropout_masks: Array<Record<string, any>>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
  parameters_used: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const trainingCurveData = useMemo(() => {
    const curves = result.training_curves;
    if (!curves || Object.keys(curves).length === 0) {
      return [];
    }

    // Try to find epochs array or create indices
    const epochs = curves.epochs ||
                   Array.from({ length: Math.max(...Object.values(curves as any).filter(Array.isArray).map((arr: any) => arr.length || 0)) }, (_, i) => i);

    if (!Array.isArray(epochs) || epochs.length === 0) {
      return [];
    }

    return epochs.map((epoch, idx) => {
      const point: Record<string, any> = { epoch: typeof epoch === 'number' ? epoch : idx };
      Object.entries(curves).forEach(([key, vals]: [string, any]) => {
        if (key !== 'epochs' && Array.isArray(vals) && idx < vals.length) {
          point[key] = parseFloat(vals[idx].toFixed(4));
        }
      });
      return point;
    });
  }, [result.training_curves]);

  return (
    <div className="space-y-6">
      {/* Training Curves */}
      {trainingCurveData.length > 0 && (
        <div className="space-y-4">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100">Training & Validation Loss</h3>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingCurveData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
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
                {Object.keys(result.training_curves || {})
                  .filter((key) => key !== 'epochs')
                  .map((key) => {
                    const isTraining = key.toLowerCase().includes('train');
                    const isValidation = key.toLowerCase().includes('val');
                    const color = isTraining ? '#3b82f6' : isValidation ? '#ef4444' : '#6b7280';
                    return (
                      <Line
                        key={key}
                        type="monotone"
                        dataKey={key}
                        stroke={color}
                        dot={false}
                        isAnimationActive={false}
                      />
                    );
                  })}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Dropout Comparison */}
      {result.dropout_comparison && result.dropout_comparison.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Dropout Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-gray-200 dark:border-gray-700">
                  <tr>
                    {result.dropout_comparison[0] &&
                      Object.keys(result.dropout_comparison[0]).map((key) => (
                        <th key={key} className="text-left py-2 px-4 text-gray-700 dark:text-gray-300 font-semibold">
                          {key}
                        </th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {result.dropout_comparison.map((row, idx) => (
                    <tr key={idx} className="border-b border-gray-200 dark:border-gray-700">
                      {Object.values(row).map((val, i) => (
                        <td key={i} className="py-2 px-4 text-gray-600 dark:text-gray-400">
                          {typeof val === 'number' ? val.toFixed(4) : String(val)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Overfitting Metrics */}
      {result.overfitting_metrics && Object.keys(result.overfitting_metrics).length > 0 && (
        <Card className="bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800">
          <CardHeader>
            <CardTitle className="text-purple-900 dark:text-purple-100">Overfitting Metrics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-purple-800 dark:text-purple-200">
            {Object.entries(result.overfitting_metrics).map(([key, val]) => (
              <p key={key}>
                <span className="font-medium">{key}:</span> {typeof val === 'number' ? val.toFixed(4) : String(val)}
              </p>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Model Info */}
      {result.model_info && Object.keys(result.model_info).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Model Architecture</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {Object.entries(result.model_info).map(([key, val]) => (
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
