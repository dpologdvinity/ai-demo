import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface A3CParams {
  num_workers: number;
  actor_lr: number;
  critic_lr: number;
  gamma: number;
  episodes_per_worker: number;
  entropy_coef: number;
  hidden_size: number;
  random_state: number;
}

interface ControlsProps {
  parameters: A3CParams;
  onChange: (name: keyof A3CParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Number of Workers */}
      <ParameterControl
        label="Number of Workers"
        description="Parallel workers for asynchronous training (2-8)"
        type="range"
        value={parameters.num_workers}
        onChange={(value) => onChange('num_workers', parseInt(String(value)))}
        min={2}
        max={8}
        step={1}
      />

      {/* Actor Learning Rate */}
      <ParameterControl
        label="Actor Learning Rate"
        description="Learning rate for actor network (0.0001-0.01)"
        type="range"
        value={parameters.actor_lr}
        onChange={(value) => onChange('actor_lr', parseFloat(String(value)))}
        min={0.0001}
        max={0.01}
        step={0.0001}
      />

      {/* Critic Learning Rate */}
      <ParameterControl
        label="Critic Learning Rate"
        description="Learning rate for critic network (0.0001-0.01)"
        type="range"
        value={parameters.critic_lr}
        onChange={(value) => onChange('critic_lr', parseFloat(String(value)))}
        min={0.0001}
        max={0.01}
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

      {/* Episodes per Worker */}
      <ParameterControl
        label="Episodes per Worker"
        description="Episodes each worker runs (50-500)"
        type="range"
        value={parameters.episodes_per_worker}
        onChange={(value) => onChange('episodes_per_worker', parseInt(String(value)))}
        min={50}
        max={500}
        step={10}
      />

      {/* Entropy Coefficient */}
      <ParameterControl
        label="Entropy Coefficient"
        description="Exploration bonus coefficient (0.0-0.1)"
        type="range"
        value={parameters.entropy_coef}
        onChange={(value) => onChange('entropy_coef', parseFloat(String(value)))}
        min={0.0}
        max={0.1}
        step={0.001}
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
          Train with {parameters.num_workers} parallel workers ({parameters.num_workers * parameters.episodes_per_worker} total episodes)
        </p>
      </div>
    </div>
  );
}
