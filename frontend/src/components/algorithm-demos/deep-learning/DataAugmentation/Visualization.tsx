import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface AugmentedImage {
  index: number;
  augmentations_applied: string[];
  description: string;
}

interface AugmentationResult {
  success: boolean;
  original_shape: number[];
  augmented_images: AugmentedImage[];
  augmentation_pipeline: Record<string, any>;
}

interface VisualizationProps {
  result: AugmentationResult;
}

export function Visualization({ result }: VisualizationProps) {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Image Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          <div>
            <p className="font-medium">Original Image Shape:</p>
            <p className="text-gray-600 dark:text-gray-400">
              {result.original_shape?.join(' × ')}
            </p>
          </div>
          <div>
            <p className="font-medium">Augmented Images Generated:</p>
            <p className="text-gray-600 dark:text-gray-400">
              {result.augmented_images?.length || 0}
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Augmentation Summary</CardTitle>
        </CardHeader>
        <CardContent className="text-xs">
          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {result.augmented_images?.map((img, idx) => (
              <div key={idx} className="p-2 bg-gray-50 dark:bg-gray-900 rounded">
                <p className="font-medium text-gray-700 dark:text-gray-300">
                  Image {img.index}
                </p>
                <p className="text-gray-600 dark:text-gray-400">{img.description}</p>
              </div>
            )) || <p className="text-gray-500">No augmented images</p>}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Augmentation Techniques</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-gray-600 dark:text-gray-400">
          <p className="mb-2">Available augmentation techniques:</p>
          <ul className="list-disc list-inside space-y-1">
            <li>Rotation: Random rotation within specified range</li>
            <li>Flip: Horizontal or vertical mirroring</li>
            <li>Brightness: Adjustment of image brightness</li>
            <li>Contrast: Enhancement or reduction of contrast</li>
            <li>Blur: Gaussian blur application</li>
            <li>Crop: Random cropping of image</li>
            <li>Scale: Resizing within bounds</li>
          </ul>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Benefits</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-gray-600 dark:text-gray-400">
          <ul className="list-disc list-inside space-y-1">
            <li>Increases training dataset size effectively</li>
            <li>Improves model generalization</li>
            <li>Reduces overfitting on small datasets</li>
            <li>Makes models robust to variations</li>
            <li>Simulates real-world variations</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
