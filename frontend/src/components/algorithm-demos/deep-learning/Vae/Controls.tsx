import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface VAEParams {
  latent_dim: number;
  encoder_hidden: number[];
  decoder_hidden: number[];
  learning_rate: number;
  epochs: number;
  beta: number;
  batch_size: number;
  random_state: number;
}

interface ControlsProps {
  parameters: VAEParams;
  onChange: (name: keyof VAEParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Latent Dimension */}
      <ParameterControl
        label="Latent Dimension"
        description="Dimensionality of the latent space (2-64)"
        type="range"
        value={parameters.latent_dim}
        onChange={(value) => onChange('latent_dim', parseInt(String(value)))}
        min={2}
        max={64}
        step={2}
      />

      {/* Learning Rate */}
      <ParameterControl
        label="Learning Rate"
        description="Learning rate for Adam optimizer (0.0001-0.01)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
        min={0.0001}
        max={0.01}
        step={0.0001}
      />

      {/* Epochs */}
      <ParameterControl
        label="Training Epochs"
        description="Number of training epochs (10-100)"
        type="range"
        value={parameters.epochs}
        onChange={(value) => onChange('epochs', parseInt(String(value)))}
        min={10}
        max={100}
        step={10}
      />

      {/* Beta (KL Weight) */}
      <ParameterControl
        label="Beta (KL Weight)"
        description="Weight for KL divergence term (0.1-10.0)"
        type="range"
        value={parameters.beta}
        onChange={(value) => onChange('beta', parseFloat(String(value)))}
        min={0.1}
        max={10.0}
        step={0.1}
      />

      {/* Batch Size */}
      <ParameterControl
        label="Batch Size"
        description="Training batch size"
        type="select"
        value={parameters.batch_size}
        onChange={(value) => onChange('batch_size', parseInt(String(value)))}
        options={[
          { label: '16', value: 16 },
          { label: '32', value: 32 },
          { label: '64', value: 64 },
          { label: '128', value: 128 },
          { label: '256', value: 256 },
          { label: '512', value: 512 },
        ]}
      />

      {/* Random State */}
      <ParameterControl
        label="Random State"
        description="Seed for reproducibility"
        type="number"
        value={parameters.random_state}
        onChange={(value) => onChange('random_state', parseInt(String(value)))}
        min={0}
        max={9999}
        step={1}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Architecture:</strong>
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Encoder: {parameters.encoder_hidden.join(' → ')} → {parameters.latent_dim}D
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200">
          Decoder: {parameters.latent_dim}D → {parameters.decoder_hidden.join(' → ')}
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-2">
          Beta controls reconstruction vs regularization trade-off. Higher beta encourages more regularized latent space.
        </p>
      </div>
    </div>
  );
}
