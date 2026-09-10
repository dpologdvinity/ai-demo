import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface BatchNormParams {
  momentum: number;
  eps: number;
  affine: boolean;
  track_running_stats: boolean;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  hidden_size: number;
  random_state: number;
}

interface ControlsProps {
  parameters: BatchNormParams;
  onChange: (name: keyof BatchNormParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      <ParameterControl
        label="Momentum"
        description="Running stats momentum (0.01-0.5)"
        type="range"
        value={parameters.momentum}
        onChange={(value) => onChange('momentum', parseFloat(String(value)))}
        min={0.01}
        max={0.5}
        step={0.05}
      />

      <ParameterControl
        label="Epsilon"
        description="Numerical stability constant (1e-8 to 1e-3)"
        type="range"
        value={parameters.eps}
        onChange={(value) => onChange('eps', parseFloat(String(value)))}
        min={1e-8}
        max={1e-3}
        step={1e-6}
      />

      <ParameterControl
        label="Affine Parameters"
        description="Use learnable scale and shift"
        type="select"
        value={parameters.affine ? 'yes' : 'no'}
        onChange={(value) => onChange('affine', value === 'yes')}
        options={[
          { label: 'Yes', value: 'yes' },
          { label: 'No', value: 'no' }
        ]}
      />

      <ParameterControl
        label="Track Running Stats"
        description="Track running mean and variance"
        type="select"
        value={parameters.track_running_stats ? 'yes' : 'no'}
        onChange={(value) => onChange('track_running_stats', value === 'yes')}
        options={[
          { label: 'Yes', value: 'yes' },
          { label: 'No', value: 'no' }
        ]}
      />

      <ParameterControl
        label="Epochs"
        description="Number of training epochs (10-200)"
        type="range"
        value={parameters.epochs}
        onChange={(value) => onChange('epochs', parseInt(String(value)))}
        min={10}
        max={200}
        step={10}
      />

      <ParameterControl
        label="Learning Rate"
        description="Learning rate for optimizer (0.0001-0.1)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
        min={0.0001}
        max={0.1}
        step={0.001}
      />

      <ParameterControl
        label="Batch Size"
        description="Training batch size"
        type="select"
        value={parameters.batch_size}
        onChange={(value) => onChange('batch_size', parseInt(String(value)))}
        options={[
          { label: '8', value: 8 },
          { label: '16', value: 16 },
          { label: '32', value: 32 },
          { label: '64', value: 64 },
          { label: '128', value: 128 }
        ]}
      />

      <ParameterControl
        label="Hidden Size"
        description="Size of hidden layers (32-256)"
        type="range"
        value={parameters.hidden_size}
        onChange={(value) => onChange('hidden_size', parseInt(String(value)))}
        min={32}
        max={256}
        step={16}
      />

      <ParameterControl
        label="Random Seed"
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
          <strong>Demo:</strong> Compares training with and without batch normalization
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Training curves will show how batch normalization accelerates convergence
        </p>
      </div>
    </div>
  );
}
