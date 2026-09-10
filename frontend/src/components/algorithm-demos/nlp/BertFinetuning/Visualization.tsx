import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface TrainingResult {
  success: boolean;
  training_history: Array<{
    epoch: number;
    train_loss: number;
    train_accuracy: number;
    val_loss?: number;
    val_accuracy?: number;
  }>;
  predictions: Array<{
    text: string;
    predicted_label: number;
    predicted_class: string;
    confidence: number;
    probabilities: number[];
  }>;
  metrics: Record<string, any>;
  execution_time_ms: number;
}

interface VisualizationProps {
  result: TrainingResult;
}

const LABEL_MAP = {
  0: 'Negative',
  1: 'Neutral',
  2: 'Positive',
};

const LABEL_COLORS = {
  0: '#ef4444',
  1: '#f59e0b',
  2: '#10b981',
};

export function Visualization({ result }: VisualizationProps) {
  const { training_history, predictions, metrics } = result;

  const trainingData = useMemo(() => {
    return training_history.map((h) => ({
      epoch: h.epoch,
      'Train Loss': h.train_loss,
      'Val Loss': h.val_loss || null,
      'Train Acc': h.train_accuracy * 100,
      'Val Acc': (h.val_accuracy || 0) * 100,
    }));
  }, [training_history]);

  const predictionSummary = useMemo(() => {
    const summary: Record<string, number> = {
      Negative: 0,
      Neutral: 0,
      Positive: 0,
    };

    predictions.forEach((p) => {
      const label = LABEL_MAP[p.predicted_label as keyof typeof LABEL_MAP];
      if (label) summary[label]++;
    });

    return [
      { name: 'Negative', count: summary.Negative, fill: LABEL_COLORS[0] },
      { name: 'Neutral', count: summary.Neutral, fill: LABEL_COLORS[1] },
      { name: 'Positive', count: summary.Positive, fill: LABEL_COLORS[2] },
    ];
  }, [predictions]);

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
                  label={{ value: 'Accuracy (%)', angle: 90, position: 'insideRight' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #ccc',
                    borderRadius: '4px',
                  }}
                />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="Train Loss" stroke="#ef4444" />
                <Line yAxisId="left" type="monotone" dataKey="Val Loss" stroke="#fbbf24" />
                <Line yAxisId="right" type="monotone" dataKey="Train Acc" stroke="#10b981" />
                <Line yAxisId="right" type="monotone" dataKey="Val Acc" stroke="#06b6d4" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Prediction Distribution */}
      {predictions.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
            Prediction Distribution
          </h4>
          <div className="h-[250px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={predictionSummary}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                <XAxis dataKey="name" className="text-gray-700 dark:text-gray-300" />
                <YAxis className="text-gray-700 dark:text-gray-300" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #ccc',
                    borderRadius: '4px',
                  }}
                />
                <Bar dataKey="count" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Sample Predictions */}
      {predictions.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
            Sample Predictions (Top 5)
          </h4>
          <div className="space-y-2">
            {predictions.slice(0, 5).map((pred, idx) => (
              <div
                key={idx}
                className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700"
              >
                <p className="text-xs font-mono text-gray-600 dark:text-gray-400 mb-2 truncate">
                  {pred.text}
                </p>
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className="px-2 py-1 text-xs font-semibold rounded text-white"
                    style={{ backgroundColor: LABEL_COLORS[pred.predicted_label as keyof typeof LABEL_COLORS] }}
                  >
                    {LABEL_MAP[pred.predicted_label as keyof typeof LABEL_MAP]}
                  </span>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  Probabilities: Neg {(pred.probabilities[0] * 100).toFixed(1)}% | Neu{' '}
                  {(pred.probabilities[1] * 100).toFixed(1)}% | Pos{' '}
                  {(pred.probabilities[2] * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Metrics Summary */}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Final Accuracy</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {(metrics.accuracy * 100).toFixed(2)}%
          </p>
        </div>
        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Final Loss</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {metrics.final_loss?.toFixed(4) || 'N/A'}
          </p>
        </div>
        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Samples Tested</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {predictions.length}
          </p>
        </div>
        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Execution Time</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {result.execution_time_ms.toFixed(0)} ms
          </p>
        </div>
      </div>
    </div>
  );
}
