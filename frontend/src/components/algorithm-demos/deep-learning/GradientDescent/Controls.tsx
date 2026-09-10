import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface GradientDescentParams {
  optimizer_type: string;
  learning_rate: number;
  momentum: number;
  iterations: number;
  compare_all: boolean;
  test_function: string;
  random_state: number;
}

interface ControlsProps {
  parameters: GradientDescentParams;
  onChange: (name: keyof GradientDescentParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Test Function */}
      <ParameterControl
        label="Test Function"
        description="Optimization landscape"
        type="select"
        value={parameters.test_function}
        onChange={(value) => onChange('test_function', value)}
        options={[
          { label: 'Rosenbrock', value: 'rosenbrock' },
          { label: 'Beale', value: 'beale' },
          { label: 'Ackley', value: 'ackley' },
          { label: 'Sphere', value: 'sphere' },
        ]}
      />

      {/* Optimizer Type */}
      <ParameterControl
        label="Optimizer"
        description="Optimization algorithm variant"
        type="select"
        value={parameters.optimizer_type}
        onChange={(value) => onChange('optimizer_type', value)}
        options={[
          { label: 'SGD', value: 'sgd' },
          { label: 'Momentum', value: 'momentum' },
          { label: 'RMSprop', value: 'rmsprop' },
          { label: 'Adam', value: 'adam' },
          { label: 'AdaGrad', value: 'adagrad' },
        ]}
      />

      {/* Learning Rate */}
      <ParameterControl
        label="Learning Rate"
        description="Step size for optimization (0.001-0.5)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
        min={0.001}
        max={0.5}
        step={0.01}
      />

      {/* Momentum */}
      <ParameterControl
        label="Momentum"
        description="Momentum coefficient for accelerated methods (0-0.99)"
        type="range"
        value={parameters.momentum}
        onChange={(value) => onChange('momentum', parseFloat(String(value)))}
        min={0}
        max={0.99}
        step={0.05}
      />

      {/* Iterations */}
      <ParameterControl
        label="Iterations"
        description="Number of optimization steps (20-500)"
        type="range"
        value={parameters.iterations}
        onChange={(value) => onChange('iterations', parseInt(String(value)))}
        min={20}
        max={500}
        step={20}
      />

      {/* Compare All */}
      <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Compare All Optimizers
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
        <p className="text-xs text-blue-900 dark:text-blue-100">
          <strong>Tip:</strong> Compare different optimizers on the same landscape to see convergence differences
        </p>
      </div>
    </div>
  );
}
