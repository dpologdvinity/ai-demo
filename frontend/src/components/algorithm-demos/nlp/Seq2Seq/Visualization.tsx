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

interface TrainingResult {
  success: boolean;
  task: string;
  sample_predictions: Array<{
    input: string;
    target: string;
    prediction: string;
    bleu_score: number;
  }>;
  training_history: Array<{
    epoch: number;
    loss: number;
    bleu: number;
  }>;
  metrics: Record<string, any>;
  vocabulary_info: Record<string, any>;
  execution_time_ms: number;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { training_history, sample_predictions, metrics, task, vocabulary_info } = result;

  const trainingData = useMemo(() => {
    return training_history.map((h) => ({
      epoch: h.epoch,
      'Loss': h.loss,
      'BLEU': h.bleu * 100,
    }));
  }, [training_history]);

  const bleuStats = useMemo(() => {
    if (sample_predictions.length === 0) return null;
    const scores = sample_predictions.map(p => p.bleu_score);
    return {
      avg: scores.reduce((a, b) => a + b, 0) / scores.length,
      max: Math.max(...scores),
      min: Math.min(...scores),
    };
  }, [sample_predictions]);

  const getTaskLabel = () => {
    switch (task) {
      case 'translation':
        return 'Machine Translation';
      case 'reversal':
        return 'Sequence Reversal';
      case 'date-conversion':
        return 'Date Format Conversion';
      default:
        return task;
    }
  };

  return (
    <div className="space-y-6">
      {/* Training History */}
      {trainingData.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
            Training History
          </h4>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                <XAxis
                  dataKey="epoch"
                  className="text-gray-700 dark:text-gray-300"
                  label={{ value: 'Epoch', position: 'insideBottomRight', offset: -10 }}
                />
                <YAxis
                  yAxisId="left"
                  className="text-gray-700 dark:text-gray-300"
                  label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  className="text-gray-700 dark:text-gray-300"
                  label={{ value: 'BLEU Score (%)', angle: 90, position: 'insideRight' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #ccc',
                    borderRadius: '4px',
                  }}
                />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="Loss"
                  stroke="#ef4444"
                  strokeWidth={2}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="BLEU"
                  stroke="#10b981"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Sample Predictions */}
      {sample_predictions.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
            Sample Predictions (Top 5)
          </h4>
          <div className="space-y-3">
            {sample_predictions.slice(0, 5).map((pred, idx) => (
              <div
                key={idx}
                className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700"
              >
                <div className="mb-3">
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1 font-semibold">Input:</p>
                  <p className="text-sm font-mono text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-900 p-2 rounded">
                    {pred.input}
                  </p>
                </div>

                <div className="mb-3">
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1 font-semibold">Target:</p>
                  <p className="text-sm font-mono text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-900 p-2 rounded">
                    {pred.target}
                  </p>
                </div>

                <div className="mb-3">
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1 font-semibold">Prediction:</p>
                  <p className="text-sm font-mono text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-900 p-2 rounded">
                    {pred.prediction}
                  </p>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600 dark:text-gray-400">BLEU Score:</span>
                  <span className="text-sm font-semibold text-blue-600 dark:text-blue-400">
                    {(pred.bleu_score * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Metrics Summary */}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Task</p>
          <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            {getTaskLabel()}
          </p>
        </div>
        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Final Loss</p>
          <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            {metrics.final_loss?.toFixed(4) || 'N/A'}
          </p>
        </div>
        {bleuStats && (
          <>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Avg BLEU</p>
              <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                {(bleuStats.avg * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Samples</p>
              <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                {sample_predictions.length}
              </p>
            </div>
          </>
        )}
      </div>

      {/* Vocabulary Info */}
      {vocabulary_info && (
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <h5 className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">
            Vocabulary Information
          </h5>
          <div className="grid grid-cols-2 gap-2 text-xs text-blue-800 dark:text-blue-200">
            {Object.entries(vocabulary_info).map(([key, value]) => (
              <div key={key}>
                <span className="font-semibold">{key}:</span> {String(value)}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
