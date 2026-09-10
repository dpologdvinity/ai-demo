import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface PPOParams {
  learning_rate: number;
  gamma: number;
  clip_epsilon: number;
  epochs: number;
  episodes: number;
  gae_lambda: number;
  batch_size: number;
  hidden_size: number;
  random_state: number;
}

interface ControlsProps {
  parameters: PPOParams;
  onChange: (name: keyof PPOParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Learning Rate */}
      <ParameterControl
        label="Learning Rate"
        description="Optimizer learning rate (0.0001-0.001)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
        min={0.0001}
        max={0.001}
        step={0.0001}
      />

      {/* Gamma (Discount Factor) */}
      <ParameterControl
        label="Discount Factor (γ)"
        description="Balance long-term vs short-term rewards (0.9-0.999)"
        type="range"
        value={parameters.gamma}
        onChange={(value) => onChange('gamma', parseFloat(String(value)))}
        min={0.9}
        max={0.999}
        step={0.01}
      />

      {/* Clip Epsilon */}
      <ParameterControl
        label="Clip Epsilon (ε)"
        description="Clipping parameter for policy updates (0.1-0.3)"
        type="range"
        value={parameters.clip_epsilon}
        onChange={(value) => onChange('clip_epsilon', parseFloat(String(value)))}
        min={0.1}
        max={0.3}
        step={0.01}
      />

      {/* GAE Lambda */}
      <ParameterControl
        label="GAE Lambda (λ)"
        description="Generalized Advantage Estimation parameter (0.9-0.99)"
        type="range"
        value={parameters.gae_lambda}
        onChange={(value) => onChange('gae_lambda', parseFloat(String(value)))}
        min={0.9}
        max={0.99}
        step={0.01}
      />

      {/* Epochs */}
      <ParameterControl
        label="PPO Epochs"
        description="Optimization epochs per update (1-10)"
        type="range"
        value={parameters.epochs}
        onChange={(value) => onChange('epochs', parseInt(String(value)))}
        min={1}
        max={10}
        step={1}
      />

      {/* Episodes */}
      <ParameterControl
        label="Training Episodes"
        description="Total number of episodes (100-2000)"
        type="range"
        value={parameters.episodes}
        onChange={(value) => onChange('episodes', parseInt(String(value)))}
        min={100}
        max={2000}
        step={50}
      />

      {/* Batch Size */}
      <ParameterControl
        label="Batch Size"
        description="Minibatch size for updates (32-256)"
        type="range"
        value={parameters.batch_size}
        onChange={(value) => onChange('batch_size', parseInt(String(value)))}
        min={32}
        max={256}
        step={32}
      />

      {/* Hidden Size */}
      <ParameterControl
        label="Network Hidden Size"
        description="Hidden layer size for actor and critic (64-256)"
        type="range"
        value={parameters.hidden_size}
        onChange={(value) => onChange('hidden_size', parseInt(String(value)))}
        min={64}
        max={256}
        step={32}
      />

      {/* Random State */}
      <ParameterControl
        label="Random State"
        description="Random seed for reproducibility"
        type="number"
        value={parameters.random_state}
        onChange={(value) => onChange('random_state', parseInt(String(value)))}
        min={0}
        max={9999}
        step={1}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Environment:</strong> CartPole-v1
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Balance a pole on a moving cart using reinforcement learning
        </p>
      </div>
    </div>
  );
}
