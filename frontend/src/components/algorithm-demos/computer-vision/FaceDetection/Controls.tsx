import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface FaceDetectionParams {
  scale_factor: number;
  min_neighbors: number;
  min_size: number;
  max_size: number | null;
  image_index: number;
}

interface ControlsProps {
  parameters: FaceDetectionParams;
  onChange: (name: keyof FaceDetectionParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Scale Factor */}
      <ParameterControl
        label="Scale Factor"
        description="Scale reduction between scans (1.05-1.3)"
        type="range"
        value={parameters.scale_factor}
        onChange={(value) => onChange('scale_factor', parseFloat(String(value)))}
        min={1.05}
        max={1.3}
        step={0.05}
      />

      {/* Min Neighbors */}
      <ParameterControl
        label="Min Neighbors"
        description="Minimum detections for confirmation (1-10)"
        type="range"
        value={parameters.min_neighbors}
        onChange={(value) => onChange('min_neighbors', parseInt(String(value)))}
        min={1}
        max={10}
        step={1}
      />

      {/* Min Size */}
      <ParameterControl
        label="Min Face Size (px)"
        description="Minimum face size in pixels (20-100)"
        type="number"
        value={parameters.min_size}
        onChange={(value) => onChange('min_size', parseInt(String(value)))}
        min={20}
        max={100}
        step={5}
      />

      {/* Max Size */}
      <ParameterControl
        label="Max Face Size (px)"
        description="Maximum face size (leave empty for no limit)"
        type="number"
        value={parameters.max_size ?? ''}
        onChange={(value) => onChange('max_size', value ? parseInt(String(value)) : null)}
        min={50}
        max={500}
        step={10}
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
          <strong>Tip:</strong> Higher min_neighbors reduces false positives but may miss faces. Lower
          scale_factor is more accurate but slower.
        </p>
      </div>
    </div>
  );
}
