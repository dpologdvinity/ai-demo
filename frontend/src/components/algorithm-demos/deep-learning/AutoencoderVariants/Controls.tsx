import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface AutoencoderVariantsParams {
  variant: string;
  latent_dim: number;
  epochs: number;
  learning_rate: number;
  noise_factor: number;
  sparsity_weight: number;
  batch_size: number;
  random_state: number;
}

interface ControlsProps {
  parameters: AutoencoderVariantsParams;
  onChange: (name: keyof AutoencoderVariantsParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Variant Selection */}
      <ParameterControl
        label="Autoencoder Variant"
        description="Choose the type of autoencoder architecture"
        type="select"
        value={parameters.variant}
        onChange={(value) => onChange('variant', value)}
        options={[
          { label: 'Vanilla', value: 'vanilla' },
          { label: 'Denoising', value: 'denoising' },
          { label: 'Sparse', value: 'sparse' },
          { label: 'Contractive', value: 'contractive' },
        ]}
      />

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

      {/* Epochs */}
      <ParameterControl
        label="Training Epochs"
        description="Number of training epochs (5-50)"
        type="range"
        value={parameters.epochs}
        onChange={(value) => onChange('epochs', parseInt(String(value)))}
        min={5}
        max={50}
        step={5}
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

      {/* Noise Factor */}
      <ParameterControl
        label="Noise Factor"
        description="Noise level for denoising variant (0.0-0.5)"
        type="range"
        value={parameters.noise_factor}
        onChange={(value) => onChange('noise_factor', parseFloat(String(value)))}
        min={0.0}
        max={0.5}
        step={0.05}
      />

      {/* Sparsity Weight */}
      <ParameterControl
        label="Sparsity Weight"
        description="L1 regularization weight for sparse variant (0.0-0.1)"
        type="range"
        value={parameters.sparsity_weight}
        onChange={(value) => onChange('sparsity_weight', parseFloat(String(value)))}
        min={0.0}
        max={0.1}
        step={0.001}
      />

      {/* Batch Size */}
      <ParameterControl
        label="Batch Size"
        description="Training batch size"
        type="select"
        value={parameters.batch_size}
        onChange={(value) => onChange('batch_size', parseInt(String(value)))}
        options={[
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
          <strong>Variants:</strong>
        </p>
        <ul className="text-xs text-blue-800 dark:text-blue-200 mt-1 space-y-1 ml-2">
          <li>• <strong>Vanilla:</strong> Standard autoencoder with MSE loss</li>
          <li>• <strong>Denoising:</strong> Learns to reconstruct clean data from noisy input</li>
          <li>• <strong>Sparse:</strong> L1 regularization encourages sparse latent representation</li>
          <li>• <strong>Contractive:</strong> Penalizes sensitivity to input perturbations</li>
        </ul>
      </div>
    </div>
  );
}
