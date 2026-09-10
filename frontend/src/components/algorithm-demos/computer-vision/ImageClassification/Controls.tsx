import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface ImageClassificationParams {
  model_name: string;
  top_k: number;
  confidence_threshold: number;
  image_index: number;
}

interface ControlsProps {
  parameters: ImageClassificationParams;
  onChange: (name: keyof ImageClassificationParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Model Selection */}
      <ParameterControl
        label="Model"
        description="Pre-trained model architecture - trades off speed vs accuracy"
        type="select"
        value={parameters.model_name}
        onChange={(value) => onChange('model_name', value)}
        options={[
          { label: 'ResNet-18 (Fast, 11M params)', value: 'resnet18' },
          { label: 'ResNet-50 (Accurate, 25M params)', value: 'resnet50' },
          { label: 'MobileNetV2 (Fastest, 3.5M params)', value: 'mobilenet_v2' },
        ]}
      />

      {/* Top K Predictions */}
      <ParameterControl
        label="Top K Predictions"
        description="Number of top predictions to return (1-10)"
        type="number"
        value={parameters.top_k}
        onChange={(value) => onChange('top_k', parseInt(String(value)))}
        min={1}
        max={10}
        step={1}
      />

      {/* Confidence Threshold */}
      <ParameterControl
        label="Confidence Threshold"
        description="Minimum confidence for predictions (higher = fewer but more confident predictions)"
        type="range"
        value={parameters.confidence_threshold}
        onChange={(value) => onChange('confidence_threshold', parseFloat(String(value)))}
        min={0.0}
        max={1.0}
        step={0.05}
      />

      {/* Image Index */}
      <ParameterControl
        label="Sample Image"
        description="Select sample image (0-14: cat, dog, airplane, car, bird, koala, apple, apple2, bus, coffee, panda, elephant, strawberry, pizza, banana)"
        type="number"
        value={parameters.image_index}
        onChange={(value) => onChange('image_index', parseInt(String(value)))}
        min={0}
        max={14}
        step={1}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Dataset:</strong> ImageNet (1000 classes)
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Pre-trained models use transfer learning for efficient classification
        </p>
      </div>
    </div>
  );
}
