import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface SARSAParams {
  environment: 'CartPole-v1' | 'FrozenLake-v1';
  learning_rate: number;
  discount_factor: number;
  epsilon: number;
  epsilon_decay: number;
  episodes: number;
  random_state: number;
}

interface ControlsProps {
  parameters: SARSAParams;
  onChange: (name: keyof SARSAParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Environment */}
      <ParameterControl
        label="Environment"
        description="Training environment"
        type="select"
        value={parameters.environment}
        onChange={(value) => onChange('environment', value)}
        options={[
          { label: 'CartPole-v1', value: 'CartPole-v1' },
          { label: 'FrozenLake-v1', value: 'FrozenLake-v1' },
        ]}
      />

      {/* Learning Rate */}
      <ParameterControl
        label="Learning Rate (Alpha)"
        description="Step size for Q-value updates (0.01-1.0)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
        min={0.01}
        max={1.0}
        step={0.05}
      />

      {/* Discount Factor */}
      <ParameterControl
        label="Discount Factor (Gamma)"
        description="Weight of future rewards (0.8-1.0)"
        type="range"
        value={parameters.discount_factor}
        onChange={(value) => onChange('discount_factor', parseFloat(String(value)))}
        min={0.8}
        max={1.0}
        step={0.05}
      />

      {/* Epsilon */}
      <ParameterControl
        label="Initial Epsilon"
        description="Initial exploration rate (0.0-1.0)"
        type="range"
        value={parameters.epsilon}
        onChange={(value) => onChange('epsilon', parseFloat(String(value)))}
        min={0.0}
        max={1.0}
        step={0.05}
      />

      {/* Epsilon Decay */}
      <ParameterControl
        label="Epsilon Decay"
        description="Decay rate per episode (0.9-1.0)"
        type="range"
        value={parameters.epsilon_decay}
        onChange={(value) => onChange('epsilon_decay', parseFloat(String(value)))}
        min={0.9}
        max={1.0}
        step={0.01}
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
          <strong>Environment:</strong> {parameters.environment}
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          {parameters.environment === 'CartPole-v1'
            ? 'Balance a pole on a moving cart'
            : 'Navigate frozen lake without falling into holes'}
        </p>
      </div>
    </div>
  );
}
