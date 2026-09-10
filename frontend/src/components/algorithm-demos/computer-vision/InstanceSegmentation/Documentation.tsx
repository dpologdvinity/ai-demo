import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Instance Segmentation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Instance Segmentation?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Instance segmentation detects individual objects and creates pixel-level masks for
            each one. Unlike semantic segmentation which assigns each pixel a class label,
            instance segmentation distinguishes between different instances of the same class.
            Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks
            on each Region of Interest.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Backbone CNN extracts image features at multiple scales</li>
            <li>Region Proposal Network (RPN) generates candidate object regions</li>
            <li>RoI Align pools features for each proposal without quantization</li>
            <li>Box head predicts class labels and refines bounding boxes</li>
            <li>Mask head generates binary mask for each instance using FCN</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Features
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Pixel-accurate masks:</strong> Each instance gets precise segmentation
            </li>
            <li>
              <strong>Multi-task learning:</strong> Detects objects and segments simultaneously
            </li>
            <li>
              <strong>RoI Align:</strong> Improves mask accuracy by avoiding spatial quantization
            </li>
            <li>
              <strong>Multi-scale detection:</strong> Handles objects of different sizes
            </li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
