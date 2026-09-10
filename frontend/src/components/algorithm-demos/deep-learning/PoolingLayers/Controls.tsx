import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface PoolingParams {
  pool_type: string;
  pool_size: number;
  stride: number;
  padding: number;
  input_size: number;
  num_channels: number;
  random_state: number;
}

interface ControlsProps {
  parameters: PoolingParams;
  onChange: (name: keyof PoolingParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      <ParameterControl
        label="Pooling Type"
        description="Type of pooling operation"
        type="select"
        value={parameters.pool_type}
        onChange={(value) => onChange('pool_type', value)}
        options={[
          { label: 'Max Pooling', value: 'max' },
          { label: 'Average Pooling', value: 'average' },
          { label: 'Global Max', value: 'global_max' },
          { label: 'Global Average', value: 'global_average' }
        ]}
      />

      <ParameterControl
        label="Pool Size"
        description="Size of pooling window (2-4)"
        type="range"
        value={parameters.pool_size}
        onChange={(value) => onChange('pool_size', parseInt(String(value)))}
        min={2}
        max={4}
        step={1}
      />

      <ParameterControl
        label="Stride"
        description="Step size for pooling (1-3)"
        type="range"
        value={parameters.stride}
        onChange={(value) => onChange('stride', parseInt(String(value)))}
        min={1}
        max={3}
        step={1}
      />

      <ParameterControl
        label="Padding"
        description="Padding around input (0-2)"
        type="range"
        value={parameters.padding}
        onChange={(value) => onChange('padding', parseInt(String(value)))}
        min={0}
        max={2}
        step={1}
      />

      <ParameterControl
        label="Input Size"
        description="Size of square input (4-32)"
        type="range"
        value={parameters.input_size}
        onChange={(value) => onChange('input_size', parseInt(String(value)))}
        min={4}
        max={32}
        step={4}
      />

      <ParameterControl
        label="Number of Channels"
        description="Input channels (1-3)"
        type="range"
        value={parameters.num_channels}
        onChange={(value) => onChange('num_channels', parseInt(String(value)))}
        min={1}
        max={3}
        step={1}
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
          <strong>Output Size:</strong> {Math.floor((parameters.input_size + 2 * parameters.padding - parameters.pool_size) / parameters.stride + 1)}×{Math.floor((parameters.input_size + 2 * parameters.padding - parameters.pool_size) / parameters.stride + 1)}
        </p>
      </div>
    </div>
  );
}
