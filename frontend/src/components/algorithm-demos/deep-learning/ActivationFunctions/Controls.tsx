import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface ActivationParams {
  function_type: string;
  alpha: number;
  input_range: [number, number];
  compare_all: boolean;
  num_points: number;
}

interface ControlsProps {
  parameters: ActivationParams;
  onChange: (name: keyof ActivationParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Function Type */}
      <ParameterControl
        label="Activation Function"
        description="Primary function to visualize"
        type="select"
        value={parameters.function_type}
        onChange={(value) => onChange('function_type', value)}
        options={[
          { label: 'ReLU', value: 'relu' },
          { label: 'Leaky ReLU', value: 'leaky_relu' },
          { label: 'Sigmoid', value: 'sigmoid' },
          { label: 'Tanh', value: 'tanh' },
          { label: 'ELU', value: 'elu' },
          { label: 'Swish', value: 'swish' },
        ]}
      />

      {/* Alpha Parameter */}
      <ParameterControl
        label="Alpha (Leaky ReLU)"
        description="Negative slope for leaky activations (0-0.3)"
        type="range"
        value={parameters.alpha}
        onChange={(value) => onChange('alpha', parseFloat(String(value)))}
        min={0}
        max={0.3}
        step={0.01}
      />

      {/* Input Range */}
      <ParameterControl
        label="Input Range (Min)"
        description="Minimum input value for visualization"
        type="number"
        value={parameters.input_range[0]}
        onChange={(value) => onChange('input_range', [parseFloat(String(value)), parameters.input_range[1]])}
        min={-100}
        max={0}
        step={1}
      />

      <ParameterControl
        label="Input Range (Max)"
        description="Maximum input value for visualization"
        type="number"
        value={parameters.input_range[1]}
        onChange={(value) => onChange('input_range', [parameters.input_range[0], parseFloat(String(value))])}
        min={0}
        max={100}
        step={1}
      />

      {/* Number of Points */}
      <ParameterControl
        label="Number of Points"
        description="Points to generate for smooth curves (50-1000)"
        type="range"
        value={parameters.num_points}
        onChange={(value) => onChange('num_points', parseInt(String(value)))}
        min={50}
        max={1000}
        step={50}
      />

      {/* Compare All */}
      <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Show All Functions
        </span>
        <button
          onClick={() => onChange('compare_all', !parameters.compare_all)}
          className={`w-12 h-6 rounded-full transition-colors ${
            parameters.compare_all
              ? 'bg-blue-500'
              : 'bg-gray-300 dark:bg-gray-600'
          }`}
        >
          <div
            className={`w-5 h-5 bg-white rounded-full shadow transition-transform ${
              parameters.compare_all ? 'translate-x-6' : 'translate-x-0.5'
            }`}
          />
        </button>
      </div>

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Info:</strong> View activation function curves, derivatives, and properties.
        </p>
      </div>
    </div>
  );
}
