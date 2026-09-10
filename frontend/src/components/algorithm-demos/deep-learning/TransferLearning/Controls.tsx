import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface TransferLearningParams {
  base_model: string;
  strategy: string;
  freeze_layers: string;
  learning_rate: number;
  epochs: number;
  batch_size: number;
  dataset: string;
  random_state: number;
}

interface ControlsProps {
  parameters: TransferLearningParams;
  onChange: (name: keyof TransferLearningParams, value: unknown) => void;
  algorithmInfo?: unknown;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      <ParameterControl
        label="Base Model"
        description="Pre-trained model to use as feature extractor"
        type="select"
        value={parameters.base_model}
        onChange={(value) => onChange('base_model', value)}
        options={[
          { label: 'ResNet-18', value: 'resnet18' },
          { label: 'ResNet-50', value: 'resnet50' },
          { label: 'MobileNet v2', value: 'mobilenet_v2' },
          { label: 'EfficientNet B0', value: 'efficientnet_b0' },
        ]}
      />

      <ParameterControl
        label="Strategy"
        description="Transfer learning approach"
        type="select"
        value={parameters.strategy}
        onChange={(value) => onChange('strategy', value)}
        options={[
          { label: 'Feature Extraction', value: 'feature_extraction' },
          { label: 'Fine-tune', value: 'fine_tune' },
          { label: 'Full Training', value: 'full_train' },
        ]}
      />

      <ParameterControl
        label="Freeze Layers"
        description="Which layers to freeze during training"
        type="select"
        value={parameters.freeze_layers}
        onChange={(value) => onChange('freeze_layers', value)}
        options={[
          { label: 'None (train all)', value: 'none' },
          { label: 'Early layers', value: 'early' },
          { label: 'Most layers', value: 'most' },
          { label: 'All but last', value: 'all_but_last' },
          { label: 'Auto', value: 'auto' },
        ]}
      />

      <ParameterControl
        label="Learning Rate"
        description="Learning rate for optimizer (0.0001-0.01)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
        min={0.0001}
        max={0.01}
        step={0.0001}
      />

      <ParameterControl
        label="Epochs"
        description="Number of training epochs (5-50)"
        type="range"
        value={parameters.epochs}
        onChange={(value) => onChange('epochs', parseInt(String(value)))}
        min={5}
        max={50}
        step={1}
      />

      <ParameterControl
        label="Batch Size"
        description="Training batch size (4-64)"
        type="range"
        value={parameters.batch_size}
        onChange={(value) => onChange('batch_size', parseInt(String(value)))}
        min={4}
        max={64}
        step={4}
      />

      <ParameterControl
        label="Dataset"
        description="Dataset for fine-tuning"
        type="select"
        value={parameters.dataset}
        onChange={(value) => onChange('dataset', value)}
        options={[
          { label: 'Flowers', value: 'flowers' },
          { label: 'Animals', value: 'animals' },
          { label: 'Food', value: 'food' },
        ]}
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
          <strong>Dataset:</strong> {parameters.dataset.charAt(0).toUpperCase() + parameters.dataset.slice(1)}
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          {parameters.strategy === 'feature_extraction'
            ? 'Feature extraction: only train the final classification layer'
            : parameters.strategy === 'fine_tune'
            ? 'Fine-tune: gradually unfreeze layers and train with low learning rate'
            : 'Full training: train all layers from scratch with higher learning rate'}
        </p>
      </div>
    </div>
  );
}
