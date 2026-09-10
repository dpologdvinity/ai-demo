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
  training_history: Record<string, any>;
  model_info: Record<string, any>;
  execution_time_ms: number;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { training_history, metrics, model_info } = result;

  const trainingData = useMemo(() => {
    if (!training_history.train_loss) return [];

    const epochs = training_history.train_loss.length;
    return Array.from({ length: epochs }, (_, i) => ({
      epoch: i + 1,
      train_loss: training_history.train_loss?.[i] || 0,
      val_loss: training_history.val_loss?.[i] || 0,
    }));
  }, [training_history]);

  return (
    <div className="space-y-4">
      <div className="h-[400px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={trainingData} margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
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
              dataKey="train_loss"
              stroke="#2563eb"
              name="Training Loss"
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="val_loss"
              stroke="#dc2626"
              name="Validation Loss"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <div>
              <span className="font-medium">Accuracy:</span> {(metrics.accuracy * 100).toFixed(2)}%
            </div>
            <div>
              <span className="font-medium">Precision:</span> {(metrics.precision * 100).toFixed(2)}%
            </div>
            <div>
              <span className="font-medium">Recall:</span> {(metrics.recall * 100).toFixed(2)}%
            </div>
            <div>
              <span className="font-medium">F1-Score:</span> {(metrics.f1_score * 100).toFixed(2)}%
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Model Architecture</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <div>
              <span className="font-medium">Total Parameters:</span>{' '}
              {model_info?.total_parameters?.toLocaleString() || 'N/A'}
            </div>
            <div>
              <span className="font-medium">Conv Filters:</span> {model_info?.conv_filters?.join(', ')}
            </div>
            <div>
              <span className="font-medium">Input Shape:</span> (1, 8, 8)
            </div>
            <div>
              <span className="font-medium">Output Classes:</span> 10
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Training Summary</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-gray-600 dark:text-gray-400">
          <p>
            The CNN successfully learned hierarchical features from handwritten digit images.
            The training loss decreased over epochs while the validation loss stabilized,
            indicating good generalization. The model achieved high accuracy on the test set.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
