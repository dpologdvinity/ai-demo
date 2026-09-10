import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface ResNetParams {
  model_variant: string;
  top_k: number;
  use_pretrained: boolean;
  image_index: number;
  random_state: number;
}

interface ControlsProps {
  parameters: ResNetParams;
  onChange: (name: keyof ResNetParams, value: unknown) => void;
  algorithmInfo?: unknown;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      <ParameterControl
        label="Model Variant"
        description="ResNet architecture variant"
        type="select"
        value={parameters.model_variant}
        onChange={(value) => onChange('model_variant', value)}
        options={[
          { label: 'ResNet-18', value: 'resnet18' },
          { label: 'ResNet-34', value: 'resnet34' },
          { label: 'ResNet-50', value: 'resnet50' },
          { label: 'ResNet-101', value: 'resnet101' },
        ]}
      />

      <ParameterControl
        label="Top-K Predictions"
        description="Number of top predictions to return (1-10)"
        type="range"
        value={parameters.top_k}
        onChange={(value) => onChange('top_k', parseInt(String(value)))}
        min={1}
        max={10}
        step={1}
      />

      <ParameterControl
        label="Use Pre-trained Weights"
        description="Use ImageNet pre-trained weights"
        type="toggle"
        value={parameters.use_pretrained}
        onChange={(value) => onChange('use_pretrained', value)}
      />

      <ParameterControl
        label="Sample Image Index"
        description="Choose a sample image from the gallery (0-9)"
        type="range"
        value={parameters.image_index}
        onChange={(value) => onChange('image_index', parseInt(String(value)))}
        min={0}
        max={9}
        step={1}
      />

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
          <strong>Dataset:</strong> ImageNet Classes
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          ResNet performs image classification on 1000 ImageNet classes. The {parameters.model_variant} variant has
          deep residual connections for improved gradient flow.
        </p>
      </div>
    </div>
  );
}
