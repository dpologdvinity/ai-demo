import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface EdgeDetectionParams {
  threshold1: number;
  threshold2: number;
  aperture_size: number;
  l2gradient: boolean;
  image_index: number;
}

interface ControlsProps {
  parameters: EdgeDetectionParams;
  onChange: (name: keyof EdgeDetectionParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Lower Threshold */}
      <ParameterControl
        label="Lower Threshold"
        description="Lower threshold for hysteresis (0-255)"
        type="range"
        value={parameters.threshold1}
        onChange={(value) => onChange('threshold1', parseInt(String(value)))}
        min={0}
        max={255}
        step={5}
      />

      {/* Upper Threshold */}
      <ParameterControl
        label="Upper Threshold"
        description="Upper threshold for hysteresis (0-255)"
        type="range"
        value={parameters.threshold2}
        onChange={(value) => onChange('threshold2', parseInt(String(value)))}
        min={0}
        max={255}
        step={5}
      />

      {/* Aperture Size */}
      <ParameterControl
        label="Aperture Size"
        description="Sobel kernel size"
        type="select"
        value={parameters.aperture_size}
        onChange={(value) => onChange('aperture_size', parseInt(String(value)))}
        options={[
          { label: '3x3 (Fast)', value: 3 },
          { label: '5x5 (Balanced)', value: 5 },
          { label: '7x7 (Smooth)', value: 7 },
        ]}
      />

      {/* L2 Gradient */}
      <ParameterControl
        label="L2 Gradient"
        description="Use L2 norm for gradient magnitude"
        type="boolean"
        value={parameters.l2gradient}
        onChange={(value) => onChange('l2gradient', value)}
      />

      {/* Image Index */}
      <ParameterControl
        label="Sample Image"
        description="Select sample image (0-7)"
        type="number"
        value={parameters.image_index}
        onChange={(value) => onChange('image_index', parseInt(String(value)))}
        min={0}
        max={7}
        step={1}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Tip:</strong> Typical ratio for thresholds is 2:1 or 3:1. Adjust thresholds to detect more or fewer edges.
        </p>
      </div>
    </div>
  );
}
