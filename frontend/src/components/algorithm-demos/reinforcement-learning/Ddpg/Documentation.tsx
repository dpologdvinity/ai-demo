import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About DDPG</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is DDPG?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            DDPG (Deep Deterministic Policy Gradient) is an off-policy actor-critic algorithm designed for continuous
            action spaces. It combines the deterministic policy gradient approach with deep learning and experience replay.
            DDPG learns a deterministic policy (mapping states directly to actions) rather than a stochastic policy,
            making it efficient for continuous control problems like robotics and autonomous systems.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Actor network learns a deterministic policy μ(s) that outputs continuous actions</li>
            <li>Critic network learns Q-function Q(s,a) to evaluate state-action pairs</li>
            <li>Experience replay stores transitions (s,a,r,s') in a replay buffer</li>
            <li>Sample random minibatches from replay buffer for training</li>
            <li>Update critic using Bellman equation with target networks</li>
            <li>Update actor using deterministic policy gradient on critic</li>
            <li>Soft update target networks using parameter τ</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Parameters
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Soft Update Coefficient (τ):</strong> Controls how fast target networks are updated.
              Small τ (0.001-0.01) provides stable learning.
            </li>
            <li>
              <strong>Actor/Critic Learning Rates:</strong> Typically actor LR is smaller than critic LR.
              Actor explores while critic provides stable targets.
            </li>
            <li>
              <strong>Replay Buffer Size:</strong> Larger buffer stores more diverse experiences.
              Helps break temporal correlations and improve stability.
            </li>
            <li>
              <strong>Batch Size:</strong> Size of minibatches sampled from replay buffer.
              Larger batches provide more stable gradient estimates.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Off-policy learning enables high sample efficiency</li>
            <li>Handles continuous action spaces without discretization</li>
            <li>Experience replay improves stability and data efficiency</li>
            <li>Target networks prevent divergence during training</li>
            <li>Deterministic policy gradient is computationally efficient</li>
            <li>Proven success on challenging robotics and control tasks</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Limitations
          </h3>
          <ul className="list-disc list-inside space-y-1 text-red-700 dark:text-red-300">
            <li>Sensitive to hyperparameter choices (learning rates, tau)</li>
            <li>Exploration via noise requires careful tuning</li>
            <li>May converge to local optima</li>
            <li>Q-value overestimation bias (like DQN)</li>
            <li>Requires significant memory for replay buffer</li>
            <li>Can be unstable during training without proper initialization</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Robot arm control and manipulation</li>
            <li>Autonomous vehicle control</li>
            <li>Drone navigation and flight control</li>
            <li>Industrial process control</li>
            <li>Continuous control benchmarks (Mujoco)</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
