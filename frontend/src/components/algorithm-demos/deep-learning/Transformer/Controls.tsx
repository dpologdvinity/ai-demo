import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface TransformerParams {
  d_model: number;
  nhead: number;
  num_layers: number;
  dim_feedforward: number;
  learning_rate: number;
  epochs: number;
  dropout: number;
  random_state: number;
}

interface ControlsProps {
  parameters: TransformerParams;
  onChange: (name: keyof TransformerParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Model Dimension */}
      <ParameterControl
        label="Model Dimension"
        description="Embedding and hidden size (64-512)"
        type="range"
        value={parameters.d_model}
        onChange={(value) => onChange('d_model', parseInt(String(value)))}
        min={64}
        max={512}
        step={64}
      />

      {/* Number of Attention Heads */}
      <ParameterControl
        label="Attention Heads"
        description="Number of parallel attention mechanisms (4 or 8)"
        type="select"
        value={parameters.nhead}
        onChange={(value) => onChange('nhead', parseInt(String(value)))}
        options={[
          { label: '4 Heads', value: 4 },
          { label: '8 Heads', value: 8 },
          { label: '16 Heads', value: 16 },
        ]}
      />

      {/* Number of Layers */}
      <ParameterControl
        label="Encoder Layers"
        description="Stacked transformer layers (1-6)"
        type="range"
        value={parameters.num_layers}
        onChange={(value) => onChange('num_layers', parseInt(String(value)))}
        min={1}
        max={6}
        step={1}
      />

      {/* Feedforward Dimension */}
      <ParameterControl
        label="Feedforward Dimension"
        description="Inner MLP dimension (256-2048)"
        type="range"
        value={parameters.dim_feedforward}
        onChange={(value) => onChange('dim_feedforward', parseInt(String(value)))}
        min={256}
        max={2048}
        step={256}
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

      {/* Dropout */}
      <ParameterControl
        label="Dropout"
        description="Regularization rate (0-0.5)"
        type="range"
        value={parameters.dropout}
        onChange={(value) => onChange('dropout', parseFloat(String(value)))}
        min={0}
        max={0.5}
        step={0.05}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Task:</strong> Sequence-to-sequence with attention mechanism
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          d_model must be divisible by nhead ({parameters.d_model} / {parameters.nhead})
        </p>
      </div>
    </div>
  );
}
