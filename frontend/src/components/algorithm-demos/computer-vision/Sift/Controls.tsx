import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface SiftParams {
  nfeatures: number;
  nOctaveLayers: number;
  contrastThreshold: number;
  edgeThreshold: number;
  sigma: number;
  image_index: number;
  match_mode: boolean;
  match_image_index: number | null;
}

interface ControlsProps {
  parameters: SiftParams;
  onChange: (name: keyof SiftParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Maximum Features */}
      <ParameterControl
        label="Maximum Features"
        description="Maximum number of features to detect (50-2000)"
        type="number"
        value={parameters.nfeatures}
        onChange={(value) => onChange('nfeatures', parseInt(String(value)))}
        min={50}
        max={2000}
        step={50}
      />

      {/* Octave Layers */}
      <ParameterControl
        label="Octave Layers"
        description="Number of scale levels per octave (1-5)"
        type="number"
        value={parameters.nOctaveLayers}
        onChange={(value) => onChange('nOctaveLayers', parseInt(String(value)))}
        min={1}
        max={5}
        step={1}
      />

      {/* Contrast Threshold */}
      <ParameterControl
        label="Contrast Threshold"
        description="Threshold for weak feature filtering (0.01-0.1)"
        type="range"
        value={parameters.contrastThreshold}
        onChange={(value) => onChange('contrastThreshold', parseFloat(String(value)))}
        min={0.01}
        max={0.1}
        step={0.01}
      />

      {/* Edge Threshold */}
      <ParameterControl
        label="Edge Threshold"
        description="Threshold for edge-like feature filtering (5-20)"
        type="range"
        value={parameters.edgeThreshold}
        onChange={(value) => onChange('edgeThreshold', parseFloat(String(value)))}
        min={5}
        max={20}
        step={1}
      />

      {/* Sigma */}
      <ParameterControl
        label="Gaussian Sigma"
        description="Sigma for first octave (0.5-3.0)"
        type="range"
        value={parameters.sigma}
        onChange={(value) => onChange('sigma', parseFloat(String(value)))}
        min={0.5}
        max={3.0}
        step={0.1}
      />

      {/* Image Index */}
      <ParameterControl
        label="Sample Image"
        description="Select sample image (0-9)"
        type="number"
        value={parameters.image_index}
        onChange={(value) => onChange('image_index', parseInt(String(value)))}
        min={0}
        max={9}
        step={1}
      />

      {/* Matching Mode */}
      <ParameterControl
        label="Enable Matching"
        description="Match features between two images"
        type="boolean"
        value={parameters.match_mode}
        onChange={(value) => onChange('match_mode', value)}
      />

      {parameters.match_mode && (
        <ParameterControl
          label="Second Image"
          description="Index of second image for matching (0-9)"
          type="number"
          value={parameters.match_image_index ?? 1}
          onChange={(value) => onChange('match_image_index', parseInt(String(value)))}
          min={0}
          max={9}
          step={1}
        />
      )}

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Tip:</strong> More features and lower thresholds find more details but are slower. Enable
          matching to find corresponding features between two images.
        </p>
      </div>
    </div>
  );
}
