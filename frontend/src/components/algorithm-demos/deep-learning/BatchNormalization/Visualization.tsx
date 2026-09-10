import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  metrics: Record<string, any>;
  training_curves: Record<string, any>;
  execution_time_ms: number;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { training_curves, metrics } = result;

  const chartData = useMemo(() => {
    if (!training_curves.epochs) return [];

    return training_curves.epochs.map((epoch: number, idx: number) => ({
      epoch,
      with_bn_loss: training_curves.with_bn_loss?.[idx] || 0,
      without_bn_loss: training_curves.without_bn_loss?.[idx] || 0,
    }));
  }, [training_curves]);

  return (
    <div className="space-y-4">
      <div className="h-[400px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
            <XAxis
              dataKey="epoch"
              label={{ value: 'Epoch', position: 'bottom', offset: 40 }}
              className="text-gray-700 dark:text-gray-300"
            />
            <YAxis
              label={{ value: 'Loss', angle: -90, position: 'left', offset: 40 }}
              className="text-gray-700 dark:text-gray-300"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #ccc',
                borderRadius: '4px',
              }}
            />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
            <Line
              type="monotone"
              dataKey="with_bn_loss"
              stroke="#4daf4a"
              name="With Batch Norm"
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="without_bn_loss"
              stroke="#e41a1c"
              name="Without Batch Norm"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Performance Comparison</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <div>
              <span className="font-medium">With BN Final Loss:</span>{' '}
              {metrics.with_bn_final_loss?.toFixed(4)}
            </div>
            <div>
              <span className="font-medium">Without BN Final Loss:</span>{' '}
              {metrics.without_bn_final_loss?.toFixed(4)}
            </div>
            <div>
              <span className="font-medium">Improvement:</span> {metrics.improvement_percent?.toFixed(2)}%
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Convergence Analysis</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <div>
              <span className="font-medium">With BN:</span>{' '}
              {metrics.with_bn_epochs_to_converge} epochs
            </div>
            <div>
              <span className="font-medium">Without BN:</span>{' '}
              {metrics.without_bn_epochs_to_converge} epochs
            </div>
            <div>
              <span className="font-medium">Speedup:</span>{' '}
              {(metrics.without_bn_epochs_to_converge / metrics.with_bn_epochs_to_converge).toFixed(2)}x
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">About This Visualization</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-gray-600 dark:text-gray-400">
          The chart shows training loss curves for models with and without batch normalization. Batch normalization typically:
          <ul className="list-disc list-inside mt-2 space-y-1">
            <li>Accelerates convergence (reaches target loss faster)</li>
            <li>Improves final model performance</li>
            <li>Allows higher learning rates</li>
            <li>Reduces sensitivity to weight initialization</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
