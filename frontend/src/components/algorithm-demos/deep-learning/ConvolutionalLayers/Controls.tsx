import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface ConvParams {
  num_filters: number;
  kernel_size: number;
  stride: number;
  padding: string;
  activation: string;
  random_state: number;
}

interface ControlsProps {
  parameters: ConvParams;
  onChange: (name: keyof ConvParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      <ParameterControl
        label="Number of Filters"
        description="Number of filters to learn (8-128)"
        type="range"
        value={parameters.num_filters}
        onChange={(value) => onChange('num_filters', parseInt(String(value)))}
        min={8}
        max={128}
        step={8}
      />

      <ParameterControl
        label="Kernel Size"
        description="Size of convolution kernel"
        type="select"
        value={parameters.kernel_size}
        onChange={(value) => onChange('kernel_size', parseInt(String(value)))}
        options={[
          { label: '3x3', value: 3 },
          { label: '5x5', value: 5 },
          { label: '7x7', value: 7 }
        ]}
      />

      <ParameterControl
        label="Stride"
        description="Step size for convolution (1-3)"
        type="range"
        value={parameters.stride}
        onChange={(value) => onChange('stride', parseInt(String(value)))}
        min={1}
        max={3}
        step={1}
      />

      <ParameterControl
        label="Padding"
        description="Padding type"
        type="select"
        value={parameters.padding}
        onChange={(value) => onChange('padding', value)}
        options={[
          { label: 'Same', value: 'same' },
          { label: 'Valid', value: 'valid' }
        ]}
      />

      <ParameterControl
        label="Activation Function"
        description="Activation after convolution"
        type="select"
        value={parameters.activation}
        onChange={(value) => onChange('activation', value)}
        options={[
          { label: 'ReLU', value: 'relu' },
          { label: 'Tanh', value: 'tanh' },
          { label: 'None', value: 'none' }
        ]}
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
          <strong>Input:</strong> 8x8 image with 1 channel
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Convolution will extract {parameters.num_filters} different features from the input
        </p>
      </div>
    </div>
  );
}
