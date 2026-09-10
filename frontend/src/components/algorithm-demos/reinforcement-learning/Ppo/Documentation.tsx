import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About PPO</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is PPO?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            PPO (Proximal Policy Optimization) is a state-of-the-art policy gradient reinforcement learning algorithm
            that balances performance with training stability. Unlike naive policy gradients, PPO uses a clipped objective
            to prevent destructively large policy updates in a single step, making it more robust and easier to tune.
            It's widely used in robotics, game AI, and continuous control tasks.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Collect rollouts from the current policy using an actor network</li>
            <li>Compute advantages using Generalized Advantage Estimation (GAE)</li>
            <li>Create minibatches from collected experience</li>
            <li>Perform multiple optimization epochs on minibatches with clipped objective</li>
            <li>Update both actor (policy) and critic (value) networks</li>
            <li>Repeat with new rollouts from updated policy</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Parameters
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Clip Epsilon (ε):</strong> Controls clipping of probability ratios. Values between 0.1-0.2 are common.
              Limits policy change magnitude per update step.
            </li>
            <li>
              <strong>GAE Lambda (λ):</strong> Balances bias-variance tradeoff in advantage estimation.
              Closer to 1.0 uses full returns (lower bias, higher variance).
            </li>
            <li>
              <strong>PPO Epochs:</strong> Number of passes over collected data. More epochs = better sample efficiency but
              more computation.
            </li>
            <li>
              <strong>Learning Rate:</strong> Controls step size for gradient updates. Smaller values = more stable but slower
              learning.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Excellent balance of stability and sample efficiency</li>
            <li>Clipped objective prevents catastrophic policy updates</li>
            <li>Works for both discrete and continuous action spaces</li>
            <li>Relatively insensitive to hyperparameter choices</li>
            <li>On-policy learning with data reuse via multiple epochs</li>
            <li>Industry-standard algorithm widely used in practice</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Limitations
          </h3>
          <ul className="list-disc list-inside space-y-1 text-red-700 dark:text-red-300">
            <li>On-policy (less sample efficient than off-policy methods like DDPG)</li>
            <li>Requires larger batch sizes for stable training</li>
            <li>Computational overhead from multiple epochs per update</li>
            <li>Can get stuck in local optima</li>
            <li>Requires careful tuning of clip epsilon value</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Robot arm control and manipulation tasks</li>
            <li>Game AI and autonomous agents</li>
            <li>Continuous control (walking, swimming, flying)</li>
            <li>Resource optimization and scheduling</li>
            <li>Autonomous navigation and path planning</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
