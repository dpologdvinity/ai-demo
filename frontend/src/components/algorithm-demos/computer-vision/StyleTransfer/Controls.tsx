import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface StyleTransferParams {
  content_image_index: number;
  style_image_index: number;
  iterations: number;
  content_weight: number;
  style_weight: number;
  learning_rate: number;
  image_size: number;
}

interface ControlsProps {
  parameters: StyleTransferParams;
  onChange: (name: keyof StyleTransferParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Content Image Index */}
      <ParameterControl
        label="Content Image"
        description="Select content image (landscapes, portraits, architecture)"
        type="number"
        value={parameters.content_image_index}
        onChange={(value) => onChange('content_image_index', parseInt(String(value)))}
        min={0}
        max={9}
        step={1}
      />

      {/* Style Image Index */}
      <ParameterControl
        label="Style Image"
        description="Select artistic style (Van Gogh, Picasso, Monet, etc.)"
        type="number"
        value={parameters.style_image_index}
        onChange={(value) => onChange('style_image_index', parseInt(String(value)))}
        min={0}
        max={9}
        step={1}
      />

      {/* Iterations */}
      <ParameterControl
        label="Optimization Steps"
        description="Number of optimization iterations (more = better quality but slower)"
        type="number"
        value={parameters.iterations}
        onChange={(value) => onChange('iterations', parseInt(String(value)))}
        min={50}
        max={1000}
        step={50}
      />

      {/* Content Weight */}
      <ParameterControl
        label="Content Weight"
        description="Weight for content preservation (higher = more content preservation)"
        type="range"
        value={parameters.content_weight}
        onChange={(value) => onChange('content_weight', parseFloat(String(value)))}
        min={0.1}
        max={10.0}
        step={0.1}
      />

      {/* Style Weight */}
      <ParameterControl
        label="Style Weight"
        description="Weight for style transfer (higher = stronger style application)"
        type="range"
        value={parameters.style_weight}
        onChange={(value) => onChange('style_weight', parseFloat(String(value)))}
        min={100000.0}
        max={10000000.0}
        step={100000.0}
      />

      {/* Learning Rate */}
      <ParameterControl
        label="Learning Rate"
        description="Optimizer learning rate (higher = faster convergence but less stable)"
        type="range"
        value={parameters.learning_rate}
        onChange={(value) => onChange('learning_rate', parseFloat(String(value)))}
        min={0.001}
        max={0.01}
        step={0.001}
      />

      {/* Image Size */}
      <ParameterControl
        label="Output Size (px)"
        description="Output image size - larger is higher quality but slower"
        type="select"
        value={parameters.image_size}
        onChange={(value) => onChange('image_size', parseInt(String(value)))}
        options={[
          { label: '256x256 (Fast)', value: 256 },
          { label: '512x512 (Balanced)', value: 512 },
          { label: '1024x1024 (High Quality)', value: 1024 },
        ]}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Tips:</strong> Start with default settings. Increase iterations for better
          results. Adjust content/style weights to balance preservation and stylization.
        </p>
      </div>
    </div>
  );
}
