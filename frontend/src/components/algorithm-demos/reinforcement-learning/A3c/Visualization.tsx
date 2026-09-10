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

  const advantages = useMemo(() => {
    const advs = visualization_data?.advantages || [];
    return advs.map((adv: number, index: number) => ({
      step: index + 1,
      advantage: parseFloat(adv.toFixed(3)),
    }));
  }, [visualization_data]);

  const entropies = useMemo(() => {
    const ents = visualization_data?.entropies || [];
    return ents.map((ent: number, index: number) => ({
      step: index + 1,
      entropy: parseFloat(ent.toFixed(4)),
    }));
  }, [visualization_data]);

  const workerRewards = useMemo(() => {
    const rewards = visualization_data?.worker_rewards || [];
    // Format worker rewards by episode
    const formatted: Record<string, any> = {};
    rewards.forEach((workerData: any, workerIdx: number) => {
      if (Array.isArray(workerData)) {
        workerData.forEach((reward: number, epIdx: number) => {
          if (!formatted[epIdx]) {
            formatted[epIdx] = { episode: epIdx + 1 };
          }
          formatted[epIdx][`worker_${workerIdx}`] = parseFloat(reward.toFixed(2));
        });
      }
    });
    return Object.values(formatted);
  }, [visualization_data]);

  return (
    <div className="space-y-6">
      {/* Episode Rewards */}
      {episodeRewards.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Average Episode Rewards
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
                  name="Avg Reward"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Worker Rewards */}
      {workerRewards.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Worker Rewards
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={workerRewards} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
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
                {visualization_data?.num_workers && Array.from({ length: visualization_data.num_workers }).map((_, i) => (
                  <Line
                    key={i}
                    type="monotone"
                    dataKey={`worker_${i}`}
                    dot={false}
                    strokeWidth={1.5}
                    name={`Worker ${i}`}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Advantages */}
      {advantages.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Average Advantages
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={advantages} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
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
                  dataKey="advantage"
                  stroke="#f59e0b"
                  dot={false}
                  strokeWidth={2}
                  name="Avg Advantage"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Entropies */}
      {entropies.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Policy Entropy
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={entropies} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
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
                  dataKey="entropy"
                  stroke="#10b981"
                  dot={false}
                  strokeWidth={2}
                  name="Policy Entropy"
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
            <p>Avg Reward: {metrics.avg_reward?.toFixed(2) ?? 'N/A'}</p>
            <p>Success Rate: {(metrics.success_rate * 100)?.toFixed(1) ?? 'N/A'}%</p>
          </div>
        </div>
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Training Metrics
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Avg Entropy: {metrics.avg_entropy?.toFixed(4) ?? 'N/A'}</p>
            <p>Workers: {visualization_data?.num_workers ?? 'N/A'}</p>
          </div>
        </div>
      </div>

      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> Episode rewards show overall learning progress. Worker rewards show per-worker
          performance. Higher entropy indicates more exploration. Advantages measure how good actions are relative to baseline.
        </p>
      </div>
    </div>
  );
}
