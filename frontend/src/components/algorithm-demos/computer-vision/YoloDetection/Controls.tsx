import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface YoloDetectionParams {
  confidence_threshold: number;
  iou_threshold: number;
  model_version: string;
  max_detections: number;
  image_index: number;
  class_filter: string;
}

interface ControlsProps {
  parameters: YoloDetectionParams;
  onChange: (name: keyof YoloDetectionParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Model Version */}
      <ParameterControl
        label="Model Version"
        description="YOLO version - trades off speed vs accuracy"
        type="select"
        value={parameters.model_version}
        onChange={(value) => onChange('model_version', value)}
        options={[
          { label: 'YOLOv8 Nano (Fastest)', value: 'yolov8n' },
          { label: 'YOLOv8 Small (Balanced)', value: 'yolov8s' },
          { label: 'YOLOv8 Medium (Most Accurate)', value: 'yolov8m' },
          { label: 'YOLOv5 Small', value: 'yolov5s' },
        ]}
      />

      {/* Confidence Threshold */}
      <ParameterControl
        label="Confidence Threshold"
        description="Minimum confidence score for detections (higher = fewer but more confident detections)"
        type="range"
        value={parameters.confidence_threshold}
        onChange={(value) => onChange('confidence_threshold', parseFloat(String(value)))}
        min={0.1}
        max={0.9}
        step={0.05}
      />

      {/* IoU Threshold */}
      <ParameterControl
        label="IoU Threshold"
        description="IoU threshold for Non-Maximum Suppression (higher = more overlapping boxes)"
        type="range"
        value={parameters.iou_threshold}
        onChange={(value) => onChange('iou_threshold', parseFloat(String(value)))}
        min={0.1}
        max={0.9}
        step={0.05}
      />

      {/* Max Detections */}
      <ParameterControl
        label="Max Detections"
        description="Maximum number of objects to detect in the image"
        type="number"
        value={parameters.max_detections}
        onChange={(value) => onChange('max_detections', parseInt(String(value)))}
        min={10}
        max={300}
        step={10}
      />

      {/* Image Index */}
      <ParameterControl
        label="Sample Image"
        description="Select sample image (0-14: diverse indoor/outdoor scenes)"
        type="number"
        value={parameters.image_index}
        onChange={(value) => onChange('image_index', parseInt(String(value)))}
        min={0}
        max={14}
        step={1}
      />

      {/* Class Filter */}
      <ParameterControl
        label="Class Filter"
        description="Filter detections by object category"
        type="select"
        value={parameters.class_filter}
        onChange={(value) => onChange('class_filter', value)}
        options={[
          { label: 'All Classes', value: 'all' },
          { label: 'Person', value: 'person' },
          { label: 'Vehicle', value: 'vehicle' },
          { label: 'Animal', value: 'animal' },
        ]}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Dataset:</strong> COCO (80 object classes)
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          YOLO performs real-time detection with speed-accuracy tradeoff via model size
        </p>
      </div>
    </div>
  );
}
