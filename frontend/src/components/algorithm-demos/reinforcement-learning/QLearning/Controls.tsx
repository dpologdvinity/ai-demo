import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface QLearningParams {
  learning_rate: number;
  discount_factor: number;
  epsilon: number;
  episodes: number;
  grid_size: number;
  random_state: number;
}

interface ControlsProps {
  parameters: QLearningParams;
  onChange: (name: keyof QLearningParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
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
        description="Weight of future rewards (0.5-0.99)"
        type="range"
        value={parameters.discount_factor}
        onChange={(value) => onChange('discount_factor', parseFloat(String(value)))}
        min={0.5}
        max={0.99}
        step={0.05}
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
        description="Number of training episodes (100-5000)"
        type="range"
        value={parameters.episodes}
        onChange={(value) => onChange('episodes', parseInt(String(value)))}
        min={100}
        max={5000}
        step={100}
      />

      {/* Grid Size */}
      <ParameterControl
        label="Grid Size"
        description="Grid world dimensions (3-10)"
        type="range"
        value={parameters.grid_size}
        onChange={(value) => onChange('grid_size', parseInt(String(value)))}
        min={3}
        max={10}
        step={1}
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
          <strong>Environment:</strong> Grid World ({parameters.grid_size}x{parameters.grid_size})
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Navigate from start to goal in a discrete grid environment
        </p>
      </div>
    </div>
  );
}
