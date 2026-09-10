import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Actor-Critic</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Actor-Critic?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Actor-Critic is a hybrid reinforcement learning architecture that combines policy
            gradient (actor) and value function (critic) methods. The actor learns the policy
            (which actions to take), while the critic learns to estimate the value function
            (how good states are). They work together to achieve better convergence and stability.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Actor selects action based on current policy</li>
            <li>Environment returns reward and next state</li>
            <li>Critic estimates value of current and next state</li>
            <li>Compute temporal difference (TD) error as feedback</li>
            <li>Actor gradient is scaled by advantage (TD error)</li>
            <li>Update both actor and critic networks</li>
            <li>Repeat until convergence</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Components
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Actor Network:</strong> Outputs probability distribution over actions.
              Updated using policy gradient scaled by advantage.
            </li>
            <li>
              <strong>Critic Network:</strong> Estimates state value. Updated using temporal
              difference error as target.
            </li>
            <li>
              <strong>Advantage:</strong> TD error showing how much better/worse than expected.
              Stabilizes actor learning.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Faster convergence than pure policy gradient</li>
            <li>Lower variance than policy gradient alone</li>
            <li>Works well with function approximation</li>
            <li>Can solve continuous action space problems</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Continuous control tasks (robot manipulation)</li>
            <li>Complex environments with large state spaces</li>
            <li>Problems requiring deep neural networks</li>
            <li>Foundation for more advanced methods (A3C, PPO, TRPO)</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
