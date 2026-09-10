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

  const policyLosses = useMemo(() => {
    const losses = visualization_data?.policy_losses || [];
    return losses.map((loss: number, index: number) => ({
      update: index + 1,
      loss: parseFloat(loss.toFixed(4)),
    }));
  }, [visualization_data]);

  const valueLosses = useMemo(() => {
    const losses = visualization_data?.value_losses || [];
    return losses.map((loss: number, index: number) => ({
      update: index + 1,
      loss: parseFloat(loss.toFixed(4)),
    }));
  }, [visualization_data]);

  const clipFractions = useMemo(() => {
    const fractions = visualization_data?.clip_fractions || [];
    return fractions.map((fraction: number, index: number) => ({
      update: index + 1,
      fraction: parseFloat((fraction * 100).toFixed(2)),
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

      {/* Policy Losses */}
      {policyLosses.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Policy Loss
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={policyLosses} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                <XAxis
                  dataKey="update"
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
                  name="Policy Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Value Losses */}
      {valueLosses.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Value Loss
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={valueLosses} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                <XAxis
                  dataKey="update"
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
                  name="Value Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Clip Fractions */}
      {clipFractions.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Clip Fraction (%)
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={clipFractions} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                <XAxis
                  dataKey="update"
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
                  dataKey="fraction"
                  stroke="#10b981"
                  dot={false}
                  strokeWidth={2}
                  name="Clip Fraction (%)"
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
            <p>Success Rate: {(metrics.success_rate * 100)?.toFixed(1) ?? 'N/A'}%</p>
          </div>
        </div>
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Training Metrics
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Avg Clip Frac: {metrics.avg_clip_fraction?.toFixed(4) ?? 'N/A'}</p>
            <p>Avg KL Div: {metrics.avg_kl_divergence?.toFixed(4) ?? 'N/A'}</p>
          </div>
        </div>
      </div>

      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> Episode rewards show learning progress. Low clip fraction indicates
          stable policy updates. Policy and value losses should decrease over time for effective learning.
        </p>
      </div>
    </div>
  );
}
