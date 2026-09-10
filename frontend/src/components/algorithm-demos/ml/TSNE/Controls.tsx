import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface TSNEParams {
  n_components: number;
  perplexity: number;
  learning_rate: number;
  n_iter: number;
  random_state: number;
}

interface ControlsProps {
  parameters: TSNEParams;
  onChange: (name: keyof TSNEParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Embedding Dimensions */}
      <ParameterControl
        label="Embedding Dimensions"
        description="Number of dimensions for the embedded space"
        type="select"
        value={parameters.n_components}
        onChange={(value) => onChange('n_components', value)}
        options={[
          { label: '2D', value: 2 },
          { label: '3D', value: 3 },
        ]}
      />

      {/* Perplexity */}
      <ParameterControl
        label="Perplexity"
        description="Balance between local and global structure (5-50)"
        type="range"
        value={parameters.perplexity}
        onChange={(value) => onChange('perplexity', parseFloat(String(value)))}
        min={5}
        max={50}
        step={5}
      />

      {/* Learning Rate */}
      <ParameterControl
        label="Learning Rate"
        description="Learning rate for gradient descent (10-1000)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
        min={10}
        max={1000}
        step={10}
      />

      {/* Number of Iterations */}
      <ParameterControl
        label="Iterations"
        description="Number of optimization iterations (250-5000)"
        type="range"
        value={parameters.n_iter}
        onChange={(value) => onChange('n_iter', parseInt(String(value)))}
        min={250}
        max={5000}
        step={250}
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
          <strong>Dataset:</strong> Handwritten Digits (1,797 samples, 64 dimensions)
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          t-SNE will reduce from 64D to {parameters.n_components}D for visualization
        </p>
      </div>
    </div>
  );
}
