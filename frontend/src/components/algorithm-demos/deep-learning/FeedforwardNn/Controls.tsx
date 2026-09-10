import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface MLPParams {
  hidden_layers: number[];
  learning_rate: number;
  epochs: number;
  batch_size: number;
  activation: string;
  dataset_name: string;
  random_state: number;
}

interface ControlsProps {
  parameters: MLPParams;
  onChange: (name: keyof MLPParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  const handleHiddenLayersChange = (value: string) => {
    try {
      const layers = JSON.parse(`[${value}]`);
      if (Array.isArray(layers) && layers.every((n) => typeof n === 'number' && n > 0)) {
        onChange('hidden_layers', layers);
      }
    } catch {
      // Invalid input, ignore
    }
  };

  return (
    <div className="space-y-4">
      {/* Hidden Layers */}
      <ParameterControl
        label="Hidden Layer Sizes"
        description="Comma-separated sizes (e.g., 64,32 for two layers)"
        type="text"
        value={parameters.hidden_layers.join(',')}
        onChange={(value) => handleHiddenLayersChange(String(value))}
      />

      {/* Activation Function */}
      <ParameterControl
        label="Activation Function"
        description="Non-linearity for hidden layers"
        type="select"
        value={parameters.activation}
        onChange={(value) => onChange('activation', value)}
        options={[
          { label: 'ReLU', value: 'relu' },
          { label: 'Tanh', value: 'tanh' },
          { label: 'Sigmoid', value: 'sigmoid' },
          { label: 'Logistic', value: 'logistic' },
        ]}
      />

      {/* Learning Rate */}
      <ParameterControl
        label="Learning Rate"
        description="Optimizer step size (0.0001-0.1)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
        min={0.0001}
        max={0.1}
        step={0.0001}
      />

      {/* Epochs */}
      <ParameterControl
        label="Epochs"
        description="Training iterations (10-500)"
        type="range"
        value={parameters.epochs}
        onChange={(value) => onChange('epochs', parseInt(String(value)))}
        min={10}
        max={500}
        step={10}
      />

      {/* Batch Size */}
      <ParameterControl
        label="Batch Size"
        description="Minibatch size for training (8-128)"
        type="range"
        value={parameters.batch_size}
        onChange={(value) => onChange('batch_size', parseInt(String(value)))}
        min={8}
        max={128}
        step={8}
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
          <strong>Dataset:</strong> Iris Classification (150 samples, 4 features, 3 classes)
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Network: Input(4) → Hidden{parameters.hidden_layers.map((l) => `(${l})`).join(' → ')} → Output(3)
        </p>
      </div>
    </div>
  );
}
