import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface RNNParams {
  hidden_size: number;
  num_layers: number;
  learning_rate: number;
  epochs: number;
  sequence_length: number;
  prediction_length: number;
}

interface ControlsProps {
  parameters: RNNParams;
  onChange: (name: keyof RNNParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Hidden Size */}
      <ParameterControl
        label="Hidden Size"
        description="Hidden state dimension (32-256)"
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
        description="Stacked RNN layers (1-5)"
        type="range"
        value={parameters.num_layers}
        onChange={(value) => onChange('num_layers', parseInt(String(value)))}
        min={1}
        max={5}
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

      {/* Prediction Length */}
      <ParameterControl
        label="Prediction Length"
        description="Steps to predict ahead (1-20)"
        type="range"
        value={parameters.prediction_length}
        onChange={(value) => onChange('prediction_length', parseInt(String(value)))}
        min={1}
        max={20}
        step={1}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Task:</strong> Time series prediction (synthetic sinusoidal data)
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Model learns patterns in {parameters.sequence_length}-step windows to predict {parameters.prediction_length} steps ahead
        </p>
      </div>
    </div>
  );
}
