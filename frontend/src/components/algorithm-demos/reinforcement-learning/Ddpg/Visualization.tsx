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
  metrics: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  parameters_used: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { visualization_data, metrics } = result;

  // Prepare data for line charts
  const episodeRewards = useMemo(() => {
    const rewards = visualization_data?.episode_rewards || [];
    return rewards.map((reward: number, index: number) => ({
      episode: index + 1,
      reward: parseFloat(reward.toFixed(2)),
    }));
  }, [visualization_data]);

  const actorLosses = useMemo(() => {
    const losses = visualization_data?.actor_losses || [];
    return losses.map((loss: number, index: number) => ({
      step: index + 1,
      loss: parseFloat(loss.toFixed(4)),
    }));
  }, [visualization_data]);

  const criticLosses = useMemo(() => {
    const losses = visualization_data?.critic_losses || [];
    return losses.map((loss: number, index: number) => ({
      step: index + 1,
      loss: parseFloat(loss.toFixed(4)),
    }));
  }, [visualization_data]);

  const qValues = useMemo(() => {
    const values = visualization_data?.q_values || [];
    return values.map((value: number, index: number) => ({
      step: index + 1,
      value: parseFloat(value.toFixed(2)),
    }));
  }, [visualization_data]);

  return (
    <div className="space-y-6">
      {/* Episode Rewards */}
      {episodeRewards.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Episode Rewards
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={episodeRewards} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                <XAxis
                  dataKey="episode"
                  className="text-gray-700 dark:text-gray-300"
                />
                <YAxis className="text-gray-700 dark:text-gray-300" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                <Line
                  type="monotone"
                  dataKey="reward"
                  stroke="#3b82f6"
                  dot={false}
                  strokeWidth={2}
                  name="Episode Reward"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Actor Loss */}
      {actorLosses.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Actor Loss
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={actorLosses} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                <XAxis
                  dataKey="step"
                  className="text-gray-700 dark:text-gray-300"
                />
                <YAxis className="text-gray-700 dark:text-gray-300" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#ef4444"
                  dot={false}
                  strokeWidth={2}
                  name="Actor Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Critic Loss */}
      {criticLosses.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Critic Loss
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={criticLosses} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                <XAxis
                  dataKey="step"
                  className="text-gray-700 dark:text-gray-300"
                />
                <YAxis className="text-gray-700 dark:text-gray-300" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#f59e0b"
                  dot={false}
                  strokeWidth={2}
                  name="Critic Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Q-Values */}
      {qValues.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Average Q-Values
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={qValues} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                <XAxis
                  dataKey="step"
                  className="text-gray-700 dark:text-gray-300"
                />
                <YAxis className="text-gray-700 dark:text-gray-300" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#10b981"
                  dot={false}
                  strokeWidth={2}
                  name="Q-Value"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Summary Stats */}
      <div className="grid grid-cols-2 gap-4 mt-6">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Performance
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Avg Reward: {metrics.avg_reward_last_100?.toFixed(2) ?? 'N/A'}</p>
            <p>Improvement: {metrics.improvement?.toFixed(2) ?? 'N/A'}</p>
          </div>
        </div>
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Training Metrics
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Convergence: {metrics.convergence?.toFixed(4) ?? 'N/A'}</p>
            <p>Stability: {metrics.stability?.toFixed(4) ?? 'N/A'}</p>
          </div>
        </div>
      </div>

      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> Episode rewards show learning progress. Actor and critic losses should decrease
          as training progresses. Q-values indicate learned action quality estimates.
        </p>
      </div>
    </div>
  );
}
