import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About DQN</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is DQN?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Deep Q-Network (DQN) is a breakthrough deep reinforcement learning algorithm that combines
            Q-learning with deep neural networks. It learns to map states to action values, enabling
            agents to solve high-dimensional control problems like playing Atari games. DQN uses
            experience replay and target networks to stabilize training.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Take action in environment and observe reward and next state</li>
            <li>Store experience (state, action, reward, next_state) in replay buffer</li>
            <li>Sample random batch from replay buffer to break temporal correlations</li>
            <li>Train Q-network to minimize temporal difference error</li>
            <li>Periodically update target network to stabilize learning</li>
            <li>Repeat until convergence or max episodes reached</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Features
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Experience Replay:</strong> Stores past experiences and trains on random samples
              to reduce correlations and improve stability.
            </li>
            <li>
              <strong>Target Network:</strong> Uses a separate network for computing target values,
              updated periodically to prevent divergence.
            </li>
            <li>
              <strong>Epsilon-Greedy:</strong> Balances exploration (random actions) and exploitation
              (best known actions).
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Game playing (Atari, board games)</li>
            <li>Robot control and navigation</li>
            <li>Resource allocation problems</li>
            <li>Autonomous driving</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
