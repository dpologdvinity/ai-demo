import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface InstanceSegmentationParams {
  confidence_threshold: number;
  model_backbone: string;
  mask_threshold: number;
  max_instances: number;
  image_index: number;
  nms_threshold: number;
}

interface ControlsProps {
  parameters: InstanceSegmentationParams;
  onChange: (name: keyof InstanceSegmentationParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Confidence Threshold */}
      <ParameterControl
        label="Confidence Threshold"
        description="Minimum confidence score for detections (higher = fewer but more confident instances)"
        type="range"
        value={parameters.confidence_threshold}
        onChange={(value) => onChange('confidence_threshold', parseFloat(String(value)))}
        min={0.1}
        max={0.9}
        step={0.05}
      />

      {/* Model Backbone */}
      <ParameterControl
        label="Model Backbone"
        description="Backbone network - trades off speed vs accuracy"
        type="select"
        value={parameters.model_backbone}
        onChange={(value) => onChange('model_backbone', value)}
        options={[
          { label: 'ResNet50 (Balanced)', value: 'resnet50' },
          { label: 'ResNet101 (More Accurate)', value: 'resnet101' },
        ]}
      />

      {/* Mask Threshold */}
      <ParameterControl
        label="Mask Threshold"
        description="Binary threshold for mask generation (higher = tighter masks)"
        type="range"
        value={parameters.mask_threshold}
        onChange={(value) => onChange('mask_threshold', parseFloat(String(value)))}
        min={0.1}
        max={0.9}
        step={0.05}
      />

      {/* Max Instances */}
      <ParameterControl
        label="Max Instances"
        description="Maximum number of instances to detect"
        type="number"
        value={parameters.max_instances}
        onChange={(value) => onChange('max_instances', parseInt(String(value)))}
        min={10}
        max={200}
        step={10}
      />

      {/* Image Index */}
      <ParameterControl
        label="Sample Image"
        description="Select sample image (0-9: diverse scenes with multiple objects)"
        type="number"
        value={parameters.image_index}
        onChange={(value) => onChange('image_index', parseInt(String(value)))}
        min={0}
        max={9}
        step={1}
      />

      {/* NMS Threshold */}
      <ParameterControl
        label="NMS Threshold"
        description="Non-Maximum Suppression IoU threshold (higher = more overlapping boxes)"
        type="range"
        value={parameters.nms_threshold}
        onChange={(value) => onChange('nms_threshold', parseFloat(String(value)))}
        min={0.1}
        max={0.9}
        step={0.05}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Dataset:</strong> COCO (80 object classes)
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Mask R-CNN provides both bounding boxes and pixel-level segmentation masks
        </p>
      </div>
    </div>
  );
}
