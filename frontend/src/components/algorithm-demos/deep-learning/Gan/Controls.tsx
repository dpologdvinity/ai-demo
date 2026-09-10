import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface GANParams {
  latent_dim: number;
  g_hidden: number;
  d_hidden: number;
  learning_rate: number;
  epochs: number;
  batch_size: number;
  random_state: number;
}

interface ControlsProps {
  parameters: GANParams;
  onChange: (name: keyof GANParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Latent Dimension */}
      <ParameterControl
        label="Latent Dimension"
        description="Noise vector dimension for generator input (32-256)"
        type="range"
        value={parameters.latent_dim}
        onChange={(value) => onChange('latent_dim', parseInt(String(value)))}
        min={32}
        max={256}
        step={16}
      />

      {/* Generator Hidden Size */}
      <ParameterControl
        label="Generator Hidden Size"
        description="Size of generator's hidden layers (128-512)"
        type="range"
        value={parameters.g_hidden}
        onChange={(value) => onChange('g_hidden', parseInt(String(value)))}
        min={128}
        max={512}
        step={32}
      />

      {/* Discriminator Hidden Size */}
      <ParameterControl
        label="Discriminator Hidden Size"
        description="Size of discriminator's hidden layers (128-512)"
        type="range"
        value={parameters.d_hidden}
        onChange={(value) => onChange('d_hidden', parseInt(String(value)))}
        min={128}
        max={512}
        step={32}
      />

      {/* Learning Rate */}
      <ParameterControl
        label="Learning Rate"
        description="Learning rate for Adam optimizer (0.00001-0.001)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
        min={0.00001}
        max={0.001}
        step={0.00001}
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
          <strong>Dataset:</strong> MNIST Digits
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Generator learns to create synthetic 8×8 digit images. Training is inherently unstable —
          try different learning rates if results look poor.
        </p>
      </div>
    </div>
  );
}
