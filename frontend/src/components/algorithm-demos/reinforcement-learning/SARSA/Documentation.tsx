import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About SARSA</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is SARSA?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            SARSA (State-Action-Reward-State-Action) is an on-policy temporal difference learning
            algorithm. Unlike Q-Learning, it learns the value of the policy being followed, making
            it more conservative during exploration. It's particularly useful when the cost of
            exploration is high, as it learns safer policies.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Initialize Q-table with zeros</li>
            <li>Select action using epsilon-greedy strategy</li>
            <li>Take action, observe reward and next state</li>
            <li>Select next action using epsilon-greedy strategy</li>
            <li>Update Q-value using: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]</li>
            <li>Move to next state and repeat</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            SARSA vs Q-Learning
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>On-policy vs Off-policy:</strong> SARSA learns from the policy being followed,
              Q-Learning learns the optimal policy regardless.
            </li>
            <li>
              <strong>Update Rule:</strong> SARSA uses actual next action, Q-Learning uses max next action.
            </li>
            <li>
              <strong>Safety:</strong> SARSA tends to find safer policies because it explores more
              cautiously.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Robot control where exploration is risky</li>
            <li>Game playing with penalties for mistakes</li>
            <li>Finance and trading applications</li>
            <li>Situations requiring conservative policies</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
