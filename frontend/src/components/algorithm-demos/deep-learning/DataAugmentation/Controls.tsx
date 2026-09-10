import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface DataAugParams {
  augmentation_types: string[];
  rotation_range: number;
  brightness_factor: number;
  num_augmented: number;
  random_state: number;
}

interface ControlsProps {
  parameters: DataAugParams;
  onChange: (name: keyof DataAugParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  const augmentationOptions = [
    { label: 'Rotation', value: 'rotation' },
    { label: 'Horizontal Flip', value: 'flip_horizontal' },
    { label: 'Vertical Flip', value: 'flip_vertical' },
    { label: 'Brightness', value: 'brightness' },
    { label: 'Contrast', value: 'contrast' },
    { label: 'Blur', value: 'blur' },
    { label: 'Crop', value: 'crop' },
    { label: 'Scale', value: 'scale' },
  ];

  return (
    <div className="space-y-4">
      <ParameterControl
        label="Augmentation Types"
        description="Select augmentation techniques to apply"
        type="select"
        value={parameters.augmentation_types?.[0] || 'rotation'}
        onChange={(value) => onChange('augmentation_types', [value])}
        options={augmentationOptions}
      />

      <ParameterControl
        label="Rotation Range"
        description="Maximum rotation in degrees (0-180)"
        type="range"
        value={parameters.rotation_range}
        onChange={(value) => onChange('rotation_range', parseFloat(String(value)))}
        min={0}
        max={180}
        step={10}
      />

      <ParameterControl
        label="Brightness Factor"
        description="Brightness adjustment (0.0-1.0)"
        type="range"
        value={parameters.brightness_factor}
        onChange={(value) => onChange('brightness_factor', parseFloat(String(value)))}
        min={0.0}
        max={1.0}
        step={0.1}
      />

      <ParameterControl
        label="Number of Augmented Copies"
        description="Number of augmented images to generate (1-16)"
        type="range"
        value={parameters.num_augmented}
        onChange={(value) => onChange('num_augmented', parseInt(String(value)))}
        min={1}
        max={16}
        step={1}
      />

      <ParameterControl
        label="Random Seed"
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
          <strong>Data Augmentation:</strong> Creates diverse training examples from single images
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Helps improve model generalization and reduces overfitting
        </p>
      </div>
    </div>
  );
}
