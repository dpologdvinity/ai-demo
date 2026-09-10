import React from 'react';
import { LineChart } from '@/components/visualizations/LineChart';

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

  // Prepare episode rewards data for line chart
  const episodeRewards = visualization_data?.episode_rewards || [];
  const chartData = episodeRewards.map((reward: number, index: number) => ({
    episode: index + 1,
    reward: reward,
  }));

  return (
    <div className="space-y-6">
      {chartData.length > 0 ? (
        <>
          <LineChart
            data={chartData}
            xKey="episode"
            yKey="reward"
            title="Episode Rewards Over Training"
            xLabel="Episode"
            yLabel="Total Reward"
            height={400}
          />

          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                Performance Metrics
              </h4>
              <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <p>
                  <span className="font-medium">Avg Reward (Last 100):</span>
                  <br />
                  {(metrics?.avg_reward_last_100 || 0).toFixed(2)}
                </p>
                <p>
                  <span className="font-medium">Success Rate:</span>
                  <br />
                  {((metrics?.success_rate || 0) * 100).toFixed(1)}%
                </p>
                <p>
                  <span className="font-medium">Max Reward:</span>
                  <br />
                  {Math.max(...episodeRewards).toFixed(2)}
                </p>
              </div>
            </div>

            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                Training Info
              </h4>
              <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <p>
                  <span className="font-medium">Episodes:</span>
                  <br />
                  {result.parameters_used.episodes}
                </p>
                <p>
                  <span className="font-medium">Grid Size:</span>
                  <br />
                  {result.parameters_used.grid_size}x{result.parameters_used.grid_size}
                </p>
                <p>
                  <span className="font-medium">Execution Time:</span>
                  <br />
                  {result.execution_time_ms.toFixed(2)} ms
                </p>
              </div>
            </div>
          </div>

          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <p className="text-sm text-blue-900 dark:text-blue-100">
              <strong>Interpretation:</strong> The chart shows how the agent learns to navigate the
              grid world over episodes. Increasing rewards indicate the agent is discovering shorter
              paths to the goal. Success rate is the percentage of episodes completed successfully.
            </p>
          </div>
        </>
      ) : (
        <div className="p-8 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg text-center">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            No visualization data available. Training may have failed.
          </p>
        </div>
      )}
    </div>
  );
}
