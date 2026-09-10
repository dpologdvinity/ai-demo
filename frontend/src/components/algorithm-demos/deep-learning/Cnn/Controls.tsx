import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface CNNParams {
  conv_filters: number[];
  kernel_size: number;
  learning_rate: number;
  epochs: number;
  dropout: number;
  batch_size: number;
  random_state?: number;
  dataset_name?: string;
}

interface ControlsProps {
  parameters: CNNParams;
  onChange: (name: keyof CNNParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      <ParameterControl
        label="Kernel Size"
        description="Size of convolution kernel (odd numbers)"
        type="select"
        value={parameters.kernel_size}
        onChange={(value) => onChange('kernel_size', parseInt(String(value)))}
        options={[
          { label: '3×3', value: 3 },
          { label: '5×5', value: 5 },
          { label: '7×7', value: 7 }
        ]}
      />

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

      <ParameterControl
        label="Dropout Rate"
        description="Regularization dropout (0.0-0.8)"
        type="range"
        value={parameters.dropout}
        onChange={(value) => onChange('dropout', parseFloat(String(value)))}
        min={0.0}
        max={0.8}
        step={0.1}
      />

      <ParameterControl
        label="Batch Size"
        description="Training batch size"
        type="select"
        value={parameters.batch_size}
        onChange={(value) => onChange('batch_size', parseInt(String(value)))}
        options={[
          { label: '8', value: 8 },
          { label: '16', value: 16 },
          { label: '32', value: 32 },
          { label: '64', value: 64 },
          { label: '128', value: 128 }
        ]}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Dataset:</strong> Handwritten Digits (8×8 images, 10 classes)
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Architecture: {parameters.conv_filters?.join('→')} conv filters + FC layers
        </p>
      </div>
    </div>
  );
}
