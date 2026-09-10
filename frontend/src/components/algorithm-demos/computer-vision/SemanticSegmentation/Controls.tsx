import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface SemanticSegmentationParams {
  num_classes: number;
  confidence_threshold: number;
  model_backbone: string;
  image_size: number;
  image_index: number;
}

interface ControlsProps {
  parameters: SemanticSegmentationParams;
  onChange: (name: keyof SemanticSegmentationParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Number of Classes */}
      <ParameterControl
        label="Number of Classes"
        description="Number of segmentation classes (PASCAL VOC uses 21 classes)"
        type="number"
        value={parameters.num_classes}
        onChange={(value) => onChange('num_classes', parseInt(String(value)))}
        min={2}
        max={150}
        step={1}
      />

      {/* Confidence Threshold */}
      <ParameterControl
        label="Confidence Threshold"
        description="Minimum confidence for predictions (higher = more conservative)"
        type="range"
        value={parameters.confidence_threshold}
        onChange={(value) => onChange('confidence_threshold', parseFloat(String(value)))}
        min={0.1}
        max={0.95}
        step={0.05}
      />

      {/* Model Backbone */}
      <ParameterControl
        label="Model Backbone"
        description="Backbone architecture - trades off speed vs accuracy"
        type="select"
        value={parameters.model_backbone}
        onChange={(value) => onChange('model_backbone', value)}
        options={[
          { label: 'ResNet50 (More Accurate)', value: 'resnet50' },
          { label: 'MobileNet (Faster)', value: 'mobilenet' },
        ]}
      />

      {/* Image Size */}
      <ParameterControl
        label="Input Image Size"
        description="Input resolution for the model"
        type="select"
        value={parameters.image_size}
        onChange={(value) => onChange('image_size', parseInt(String(value)))}
        options={[
          { label: '256x256 (Fastest)', value: 256 },
          { label: '512x512 (Balanced)', value: 512 },
          { label: '1024x1024 (Best Quality)', value: 1024 },
        ]}
      />

      {/* Image Index */}
      <ParameterControl
        label="Sample Image"
        description="Select sample image (0=dog portrait, 1=dog on grass, 2=street scene)"
        type="number"
        value={parameters.image_index}
        onChange={(value) => onChange('image_index', parseInt(String(value)))}
        min={0}
        max={2}
        step={1}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Dataset:</strong> PASCAL VOC 2012 (21 classes including person, car, dog, etc.)
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Semantic segmentation assigns a class label to every pixel in the image
        </p>
      </div>
    </div>
  );
}
