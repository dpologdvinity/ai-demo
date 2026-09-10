import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface OpticalFlowParams {
  method: string;
  pyr_scale: number;
  levels: number;
  winsize: number;
  iterations: number;
  image_pair_index: number;
}

interface ControlsProps {
  parameters: OpticalFlowParams;
  onChange: (name: keyof OpticalFlowParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Method */}
      <ParameterControl
        label="Flow Algorithm"
        description="Dense vs Sparse flow estimation"
        type="select"
        value={parameters.method}
        onChange={(value) => onChange('method', String(value))}
        options={[
          { label: 'Farneback (Dense)', value: 'farneback' },
          { label: 'Lucas-Kanade (Sparse)', value: 'lucas-kanade' },
        ]}
      />

      {/* Pyramid Scale */}
      <ParameterControl
        label="Pyramid Scale"
        description="Scale reduction per pyramid level (0.3-0.9)"
        type="range"
        value={parameters.pyr_scale}
        onChange={(value) => onChange('pyr_scale', parseFloat(String(value)))}
        min={0.3}
        max={0.9}
        step={0.1}
      />

      {/* Levels */}
      <ParameterControl
        label="Pyramid Levels"
        description="Number of pyramid levels (1-5)"
        type="number"
        value={parameters.levels}
        onChange={(value) => onChange('levels', parseInt(String(value)))}
        min={1}
        max={5}
        step={1}
      />

      {/* Window Size */}
      <ParameterControl
        label="Window Size"
        description="Flow calculation window size (5-50, must be odd)"
        type="number"
        value={parameters.winsize}
        onChange={(value) => {
          let val = parseInt(String(value));
          if (val % 2 === 0) val += 1;
          onChange('winsize', val);
        }}
        min={5}
        max={50}
        step={2}
      />

      {/* Iterations */}
      <ParameterControl
        label="Iterations"
        description="Iterations per pyramid level (1-10)"
        type="number"
        value={parameters.iterations}
        onChange={(value) => onChange('iterations', parseInt(String(value)))}
        min={1}
        max={10}
        step={1}
      />

      {/* Image Pair Index */}
      <ParameterControl
        label="Sample Image Pair"
        description="Select image pair (0-2)"
        type="number"
        value={parameters.image_pair_index}
        onChange={(value) => onChange('image_pair_index', parseInt(String(value)))}
        min={0}
        max={2}
        step={1}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Tip:</strong> More pyramid levels detect larger motions. Larger window sizes are more
          robust to noise but less precise. Farneback gives dense flow; Lucas-Kanade gives sparse flow.
        </p>
      </div>
    </div>
  );
}
