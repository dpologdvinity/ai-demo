import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface AutoencoderParams {
  latent_dim: number;
  hidden_dim: number;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  random_state: number;
}

interface ControlsProps {
  parameters: AutoencoderParams;
  onChange: (name: keyof AutoencoderParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Latent Dimension */}
      <ParameterControl
        label="Latent Dimension"
        description="Dimensionality of the compressed representation (2-128)"
        type="range"
        value={parameters.latent_dim}
        onChange={(value) => onChange('latent_dim', parseInt(String(value)))}
        min={2}
        max={128}
        step={2}
      />

      {/* Hidden Dimension */}
      <ParameterControl
        label="Hidden Layer Size"
        description="Size of hidden layers in encoder and decoder (64-512)"
        type="range"
        value={parameters.hidden_dim}
        onChange={(value) => onChange('hidden_dim', parseInt(String(value)))}
        min={64}
        max={512}
        step={32}
      />

      {/* Epochs */}
      <ParameterControl
        label="Training Epochs"
        description="Number of training epochs (10-200)"
        type="range"
        value={parameters.epochs}
        onChange={(value) => onChange('epochs', parseInt(String(value)))}
        min={10}
        max={200}
        step={10}
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
          <strong>Dataset:</strong> MNIST Digits (1,797 samples, 64 dimensions)
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          The autoencoder will compress 64D images to {parameters.latent_dim}D and reconstruct them.
        </p>
      </div>
    </div>
  );
}
