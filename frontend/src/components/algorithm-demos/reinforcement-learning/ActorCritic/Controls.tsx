import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface ActorCriticParams {
  actor_lr: number;
  critic_lr: number;
  gamma: number;
  episodes: number;
  hidden_size: number;
  random_state: number;
}

interface ControlsProps {
  parameters: ActorCriticParams;
  onChange: (name: keyof ActorCriticParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Actor Learning Rate */}
      <ParameterControl
        label="Actor Learning Rate"
        description="Policy network learning rate (0.0001-0.01)"
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
        description="Value function learning rate (0.0001-0.01)"
        type="range"
        value={parameters.critic_lr}
        onChange={(value) => onChange('critic_lr', parseFloat(String(value)))}
        min={0.0001}
        max={0.01}
        step={0.0001}
      />

      {/* Gamma (Discount Factor) */}
      <ParameterControl
        label="Discount Factor (Gamma)"
        description="Weight of future rewards (0.9-0.999)"
        type="range"
        value={parameters.gamma}
        onChange={(value) => onChange('gamma', parseFloat(String(value)))}
        min={0.9}
        max={0.999}
        step={0.01}
      />

      {/* Episodes */}
      <ParameterControl
        label="Episodes"
        description="Number of training episodes (100-3000)"
        type="range"
        value={parameters.episodes}
        onChange={(value) => onChange('episodes', parseInt(String(value)))}
        min={100}
        max={3000}
        step={100}
      />

      {/* Hidden Size */}
      <ParameterControl
        label="Hidden Layer Size"
        description="Neural network hidden layer dimensions (64-256)"
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
          Actor and Critic networks learn cooperatively to balance the pole
        </p>
      </div>
    </div>
  );
}
