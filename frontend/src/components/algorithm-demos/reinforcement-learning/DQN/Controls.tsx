import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface DQNParams {
  learning_rate: number;
  gamma: number;
  epsilon: number;
  episodes: number;
  replay_buffer_size: number;
  batch_size: number;
  random_state: number;
}

interface ControlsProps {
  parameters: DQNParams;
  onChange: (name: keyof DQNParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Learning Rate */}
      <ParameterControl
        label="Learning Rate"
        description="Neural network learning rate (0.0001-0.01)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
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

      {/* Epsilon */}
      <ParameterControl
        label="Exploration Rate (Epsilon)"
        description="Probability of random action (0.0-1.0)"
        type="range"
        value={parameters.epsilon}
        onChange={(value) => onChange('epsilon', parseFloat(String(value)))}
        min={0.0}
        max={1.0}
        step={0.05}
      />

      {/* Episodes */}
      <ParameterControl
        label="Episodes"
        description="Number of training episodes (100-2000)"
        type="range"
        value={parameters.episodes}
        onChange={(value) => onChange('episodes', parseInt(String(value)))}
        min={100}
        max={2000}
        step={100}
      />

      {/* Replay Buffer Size */}
      <ParameterControl
        label="Replay Buffer Size"
        description="Memory buffer for experience replay (1000-50000)"
        type="range"
        value={parameters.replay_buffer_size}
        onChange={(value) => onChange('replay_buffer_size', parseInt(String(value)))}
        min={1000}
        max={50000}
        step={5000}
      />

      {/* Batch Size */}
      <ParameterControl
        label="Batch Size"
        description="Training batch size (16-128)"
        type="range"
        value={parameters.batch_size}
        onChange={(value) => onChange('batch_size', parseInt(String(value)))}
        min={16}
        max={128}
        step={16}
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
          Train an agent to balance a pole on a moving cart
        </p>
      </div>
    </div>
  );
}
