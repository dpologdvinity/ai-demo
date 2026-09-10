import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About A3C</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is A3C?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            A3C (Asynchronous Advantage Actor-Critic) is a parallel reinforcement learning algorithm that trains multiple
            workers asynchronously to explore the environment and update a shared global model. Each worker operates
            independently with its own environment copy, collecting diverse experiences that naturally decorrelate data.
            This approach provides faster training and better exploration than sequential methods.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Initialize shared global actor-critic network</li>
            <li>Launch multiple workers, each with local model copies</li>
            <li>Each worker collects trajectory data independently</li>
            <li>After each episode, workers compute gradients</li>
            <li>Asynchronously apply gradients to shared global model</li>
            <li>Workers fetch updated parameters from global model</li>
            <li>Repeat with different experiences from each worker</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Parameters
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Number of Workers:</strong> More workers enable parallel training but require more CPU resources.
              Typical range 4-16 workers.
            </li>
            <li>
              <strong>Entropy Coefficient:</strong> Controls exploration bonus. Higher values encourage more exploration
              by adding entropy regularization to the objective.
            </li>
            <li>
              <strong>Learning Rates:</strong> Actor and critic learn at potentially different rates.
              Actor learns policy, critic learns value baseline.
            </li>
            <li>
              <strong>Episodes per Worker:</strong> Total training time = workers × episodes_per_worker.
              Longer training improves policy but increases computation.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Parallel training significantly speeds up learning</li>
            <li>Natural experience decorrelation through asynchronous workers</li>
            <li>Advantage estimation reduces variance in policy gradients</li>
            <li>Entropy regularization encourages exploration</li>
            <li>Works for both discrete and continuous action spaces</li>
            <li>More stable than vanilla policy gradient methods</li>
            <li>No need for experience replay buffer (saves memory)</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Limitations
          </h3>
          <ul className="list-disc list-inside space-y-1 text-red-700 dark:text-red-300">
            <li>Requires multiple CPU cores for effective speedup</li>
            <li>Asynchronous updates can lead to stale gradients</li>
            <li>Hyperparameter tuning can be challenging</li>
            <li>May not fully utilize GPU acceleration</li>
            <li>Workers can temporarily learn conflicting policies</li>
            <li>Requires careful gradient clipping to prevent instability</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Real-time game AI and interactive agents</li>
            <li>Robot control and navigation</li>
            <li>Continuous control tasks with fast iteration</li>
            <li>Exploration in large state spaces</li>
            <li>Multi-agent learning scenarios</li>
            <li>CPU-efficient distributed training</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
