import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface DDPGParams {
  actor_lr: number;
  critic_lr: number;
  gamma: number;
  tau: number;
  episodes: number;
  buffer_size: number;
  batch_size: number;
  random_state: number;
}

interface ControlsProps {
  parameters: DDPGParams;
  onChange: (name: keyof DDPGParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Actor Learning Rate */}
      <ParameterControl
        label="Actor Learning Rate"
        description="Learning rate for actor network (0.00001-0.001)"
        type="range"
        value={parameters.actor_lr}
        onChange={(value) => onChange('actor_lr', parseFloat(String(value)))}
        min={0.00001}
        max={0.001}
        step={0.00001}
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

      {/* Tau (Soft Update Coefficient) */}
      <ParameterControl
        label="Soft Update Coefficient (τ)"
        description="Target network update rate (0.001-0.01)"
        type="range"
        value={parameters.tau}
        onChange={(value) => onChange('tau', parseFloat(String(value)))}
        min={0.001}
        max={0.01}
        step={0.001}
      />

      {/* Episodes */}
      <ParameterControl
        label="Training Episodes"
        description="Total number of episodes (50-500)"
        type="range"
        value={parameters.episodes}
        onChange={(value) => onChange('episodes', parseInt(String(value)))}
        min={50}
        max={500}
        step={10}
      />

      {/* Buffer Size */}
      <ParameterControl
        label="Replay Buffer Size"
        description="Experience replay buffer capacity (10000-1000000)"
        type="range"
        value={parameters.buffer_size}
        onChange={(value) => onChange('buffer_size', parseInt(String(value)))}
        min={10000}
        max={1000000}
        step={10000}
      />

      {/* Batch Size */}
      <ParameterControl
        label="Batch Size"
        description="Minibatch size for training (32-256)"
        type="range"
        value={parameters.batch_size}
        onChange={(value) => onChange('batch_size', parseInt(String(value)))}
        min={32}
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
          <strong>Environment:</strong> Pendulum-v1
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Swing a pendulum to upright position using continuous control actions
        </p>
      </div>
    </div>
  );
}
