import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface LearningRateParams {
  schedule_type: string;
  initial_lr: number;
  step_size: number;
  gamma: number;
  epochs: number;
  min_lr: number;
  max_lr: number;
  patience: number;
  random_state: number;
}

interface ControlsProps {
  parameters: LearningRateParams;
  onChange: (name: keyof LearningRateParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Schedule Type */}
      <ParameterControl
        label="Schedule Type"
        description="Learning rate scheduling strategy"
        type="select"
        value={parameters.schedule_type}
        onChange={(value) => onChange('schedule_type', value)}
        options={[
          { label: 'Step Decay', value: 'step' },
          { label: 'Exponential Decay', value: 'exponential' },
          { label: 'Cosine Annealing', value: 'cosine' },
          { label: 'Reduce on Plateau', value: 'reduce_on_plateau' },
          { label: 'Cyclic LR', value: 'cyclic' },
        ]}
      />

      {/* Initial Learning Rate */}
      <ParameterControl
        label="Initial Learning Rate"
        description="Starting learning rate (0.001-1.0)"
        type="range"
        value={parameters.initial_lr}
        onChange={(value) => onChange('initial_lr', parseFloat(String(value)))}
        min={0.001}
        max={1.0}
        step={0.01}
      />

      {/* Step Size */}
      <ParameterControl
        label="Step Size"
        description="Epochs between decay (1-50)"
        type="range"
        value={parameters.step_size}
        onChange={(value) => onChange('step_size', parseInt(String(value)))}
        min={1}
        max={50}
        step={1}
      />

      {/* Gamma */}
      <ParameterControl
        label="Decay Factor (γ)"
        description="Multiplicative factor for decay (0.01-0.9)"
        type="range"
        value={parameters.gamma}
        onChange={(value) => onChange('gamma', parseFloat(String(value)))}
        min={0.01}
        max={0.9}
        step={0.05}
      />

      {/* Epochs */}
      <ParameterControl
        label="Number of Epochs"
        description="Training duration (20-300)"
        type="range"
        value={parameters.epochs}
        onChange={(value) => onChange('epochs', parseInt(String(value)))}
        min={20}
        max={300}
        step={20}
      />

      {/* Min LR */}
      <ParameterControl
        label="Minimum Learning Rate"
        description="Lower bound for LR (0-0.1)"
        type="range"
        value={parameters.min_lr}
        onChange={(value) => onChange('min_lr', parseFloat(String(value)))}
        min={0}
        max={0.1}
        step={0.001}
      />

      {/* Max LR */}
      <ParameterControl
        label="Maximum Learning Rate"
        description="Upper bound for LR (0.01-2.0)"
        type="range"
        value={parameters.max_lr}
        onChange={(value) => onChange('max_lr', parseFloat(String(value)))}
        min={0.01}
        max={2.0}
        step={0.05}
      />

      {/* Patience */}
      <ParameterControl
        label="Patience"
        description="Epochs to wait before reducing (1-20)"
        type="range"
        value={parameters.patience}
        onChange={(value) => onChange('patience', parseInt(String(value)))}
        min={1}
        max={20}
        step={1}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-xs text-blue-900 dark:text-blue-100">
          <strong>Info:</strong> Compare how different schedules affect training convergence and final loss
        </p>
      </div>
    </div>
  );
}
