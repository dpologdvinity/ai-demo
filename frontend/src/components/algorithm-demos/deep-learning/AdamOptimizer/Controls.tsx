import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface AdamParams {
  learning_rate: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  compare_optimizers: boolean;
  max_iterations: number;
  function_type: string;
  random_state: number;
}

interface ControlsProps {
  parameters: AdamParams;
  onChange: (name: keyof AdamParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Test Function */}
      <ParameterControl
        label="Test Function"
        description="Optimization landscape to navigate"
        type="select"
        value={parameters.function_type}
        onChange={(value) => onChange('function_type', value)}
        options={[
          { label: 'Rosenbrock', value: 'rosenbrock' },
          { label: 'Beale', value: 'beale' },
          { label: 'Himmelblau', value: 'himmelblau' },
        ]}
      />

      {/* Learning Rate */}
      <ParameterControl
        label="Learning Rate (α)"
        description="Initial step size for optimization (0.0001-0.1)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
        min={0.0001}
        max={0.1}
        step={0.0001}
      />

      {/* Beta1 */}
      <ParameterControl
        label="Beta1 (First Moment)"
        description="Exponential decay for first moment estimates (0.5-0.99)"
        type="range"
        value={parameters.beta1}
        onChange={(value) => onChange('beta1', parseFloat(String(value)))}
        min={0.5}
        max={0.99}
        step={0.01}
      />

      {/* Beta2 */}
      <ParameterControl
        label="Beta2 (Second Moment)"
        description="Exponential decay for second moment estimates (0.9-0.9999)"
        type="range"
        value={parameters.beta2}
        onChange={(value) => onChange('beta2', parseFloat(String(value)))}
        min={0.9}
        max={0.9999}
        step={0.001}
      />

      {/* Max Iterations */}
      <ParameterControl
        label="Max Iterations"
        description="Number of optimization steps (50-500)"
        type="range"
        value={parameters.max_iterations}
        onChange={(value) => onChange('max_iterations', parseInt(String(value)))}
        min={50}
        max={500}
        step={25}
      />

      {/* Compare Optimizers */}
      <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Compare with Other Optimizers
        </span>
        <button
          onClick={() => onChange('compare_optimizers', !parameters.compare_optimizers)}
          className={`w-12 h-6 rounded-full transition-colors ${
            parameters.compare_optimizers
              ? 'bg-blue-500'
              : 'bg-gray-300 dark:bg-gray-600'
          }`}
        >
          <div
            className={`w-5 h-5 bg-white rounded-full shadow transition-transform ${
              parameters.compare_optimizers ? 'translate-x-6' : 'translate-x-0.5'
            }`}
          />
        </button>
      </div>

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-xs text-blue-900 dark:text-blue-100">
          <strong>Adam:</strong> Combines advantages of AdaGrad and RMSprop with momentum
        </p>
      </div>
    </div>
  );
}
