import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Q-Learning</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Q-Learning?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Q-Learning is a model-free reinforcement learning algorithm that learns the value of
            actions in states without requiring a model of the environment. It maintains a Q-table
            that maps state-action pairs to their expected values, enabling the agent to learn optimal
            policies through trial and error in discrete domains.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Initialize Q-table with zeros for all state-action pairs</li>
            <li>For each episode, select action using epsilon-greedy strategy</li>
            <li>Take action, observe reward and next state</li>
            <li>Update Q-value using: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]</li>
            <li>Repeat until episode ends or convergence</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Parameters
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Learning Rate (α):</strong> Controls how much new information overrides old.
              Higher values mean faster learning but less stability.
            </li>
            <li>
              <strong>Discount Factor (γ):</strong> Balances immediate and future rewards. Higher
              values make future rewards more important.
            </li>
            <li>
              <strong>Epsilon (ε):</strong> Probability of taking random actions for exploration
              versus exploiting known best actions.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Grid world navigation and path planning</li>
            <li>Discrete game playing</li>
            <li>Robot navigation in structured environments</li>
            <li>Learning optimal control policies</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
