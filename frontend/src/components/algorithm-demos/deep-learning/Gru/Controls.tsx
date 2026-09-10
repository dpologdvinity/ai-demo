import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface GRUParams {
  hidden_size: number;
  num_layers: number;
  learning_rate: number;
  epochs: number;
  sequence_length: number;
  dropout: number;
  batch_size: number;
  n_samples: number;
  random_state: number;
}

interface ControlsProps {
  parameters: GRUParams;
  onChange: (name: keyof GRUParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Hidden Size */}
      <ParameterControl
        label="Hidden Size"
        description="Hidden layer dimension (32-256)"
        type="range"
        value={parameters.hidden_size}
        onChange={(value) => onChange('hidden_size', parseInt(String(value)))}
        min={32}
        max={256}
        step={32}
      />

      {/* Number of Layers */}
      <ParameterControl
        label="Number of Layers"
        description="Stacked GRU layers (1-3)"
        type="range"
        value={parameters.num_layers}
        onChange={(value) => onChange('num_layers', parseInt(String(value)))}
        min={1}
        max={3}
        step={1}
      />

      {/* Learning Rate */}
      <ParameterControl
        label="Learning Rate"
        description="Optimizer step size (0.0001-0.01)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
        min={0.0001}
        max={0.01}
        step={0.0001}
      />

      {/* Epochs */}
      <ParameterControl
        label="Epochs"
        description="Training iterations (10-200)"
        type="range"
        value={parameters.epochs}
        onChange={(value) => onChange('epochs', parseInt(String(value)))}
        min={10}
        max={200}
        step={10}
      />

      {/* Sequence Length */}
      <ParameterControl
        label="Sequence Length"
        description="Input sequence length (5-50)"
        type="range"
        value={parameters.sequence_length}
        onChange={(value) => onChange('sequence_length', parseInt(String(value)))}
        min={5}
        max={50}
        step={5}
      />

      {/* Dropout */}
      <ParameterControl
        label="Dropout"
        description="Dropout rate for regularization (0-0.5)"
        type="range"
        value={parameters.dropout}
        onChange={(value) => onChange('dropout', parseFloat(String(value)))}
        min={0}
        max={0.5}
        step={0.1}
      />

      {/* Batch Size */}
      <ParameterControl
        label="Batch Size"
        description="Training batch size (16-128)"
        type="range"
        value={parameters.batch_size}
        onChange={(value) => onChange('batch_size', parseInt(String(value)))}
        min={16}
        max={128}
        step={16}
      />

      {/* Number of Samples */}
      <ParameterControl
        label="Samples"
        description="Total time steps (500-5000)"
        type="range"
        value={parameters.n_samples}
        onChange={(value) => onChange('n_samples', parseInt(String(value)))}
        min={500}
        max={5000}
        step={500}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Task:</strong> Time series forecasting with GRU gates
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          GRU is simpler than LSTM with update and reset gates
        </p>
      </div>
    </div>
  );
}
