import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface DropoutParams {
  dropout_rate: number;
  apply_to_layers: string[];
  training_epochs: number;
  hidden_layers: number[];
  learning_rate: number;
  batch_size: number;
  dataset_name: string;
  random_state: number;
}

interface ControlsProps {
  parameters: DropoutParams;
  onChange: (name: keyof DropoutParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Dropout Rate */}
      <ParameterControl
        label="Dropout Rate"
        description="Probability of dropping neurons (0-0.9)"
        type="range"
        value={parameters.dropout_rate}
        onChange={(value) => onChange('dropout_rate', parseFloat(String(value)))}
        min={0}
        max={0.9}
        step={0.05}
      />

      {/* Training Epochs */}
      <ParameterControl
        label="Training Epochs"
        description="Number of epochs to train (20-300)"
        type="range"
        value={parameters.training_epochs}
        onChange={(value) => onChange('training_epochs', parseInt(String(value)))}
        min={20}
        max={300}
        step={20}
      />

      {/* Learning Rate */}
      <ParameterControl
        label="Learning Rate"
        description="Optimizer learning rate (0.0001-0.1)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
        min={0.0001}
        max={0.1}
        step={0.001}
      />

      {/* Batch Size */}
      <ParameterControl
        label="Batch Size"
        description="Minibatch size for training (8-128)"
        type="range"
        value={parameters.batch_size}
        onChange={(value) => onChange('batch_size', parseInt(String(value)))}
        min={8}
        max={128}
        step={8}
      />

      {/* Apply to Hidden1 */}
      <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Apply to Hidden Layer 1
        </span>
        <button
          onClick={() => {
            const newLayers = parameters.apply_to_layers.includes('hidden1')
              ? parameters.apply_to_layers.filter((l) => l !== 'hidden1')
              : [...parameters.apply_to_layers, 'hidden1'];
            onChange('apply_to_layers', newLayers);
          }}
          className={`w-12 h-6 rounded-full transition-colors ${
            parameters.apply_to_layers.includes('hidden1')
              ? 'bg-blue-500'
              : 'bg-gray-300 dark:bg-gray-600'
          }`}
        >
          <div
            className={`w-5 h-5 bg-white rounded-full shadow transition-transform ${
              parameters.apply_to_layers.includes('hidden1') ? 'translate-x-6' : 'translate-x-0.5'
            }`}
          />
        </button>
      </div>

      {/* Apply to Hidden2 */}
      <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Apply to Hidden Layer 2
        </span>
        <button
          onClick={() => {
            const newLayers = parameters.apply_to_layers.includes('hidden2')
              ? parameters.apply_to_layers.filter((l) => l !== 'hidden2')
              : [...parameters.apply_to_layers, 'hidden2'];
            onChange('apply_to_layers', newLayers);
          }}
          className={`w-12 h-6 rounded-full transition-colors ${
            parameters.apply_to_layers.includes('hidden2')
              ? 'bg-blue-500'
              : 'bg-gray-300 dark:bg-gray-600'
          }`}
        >
          <div
            className={`w-5 h-5 bg-white rounded-full shadow transition-transform ${
              parameters.apply_to_layers.includes('hidden2') ? 'translate-x-6' : 'translate-x-0.5'
            }`}
          />
        </button>
      </div>

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-xs text-blue-900 dark:text-blue-100">
          <strong>Info:</strong> Compare training/validation loss to see dropout reducing overfitting
        </p>
      </div>
    </div>
  );
}
