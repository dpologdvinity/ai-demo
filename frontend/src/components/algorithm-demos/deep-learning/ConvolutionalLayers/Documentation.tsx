import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Convolutional Layers</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What are Convolutional Layers?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Convolutional layers apply learned filters to input images to extract spatial features.
            Each filter detects specific patterns (edges, textures, shapes), and stacking multiple filters
            learns increasingly complex hierarchical features.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How Convolution Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Position filter at location in input image</li>
            <li>Multiply filter weights by overlapping input values</li>
            <li>Sum products to get single output value</li>
            <li>Slide filter across entire input (stride determines step size)</li>
            <li>Create feature map from all output values</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Concepts
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Kernel/Filter:</strong> Small weight matrix that slides across input
            </li>
            <li>
              <strong>Stride:</strong> Number of pixels the filter moves at each step
            </li>
            <li>
              <strong>Padding:</strong> Zeros added around input to maintain dimensions
            </li>
            <li>
              <strong>Feature Map:</strong> Output of applying filter to input
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Weight sharing reduces parameters compared to fully connected layers</li>
            <li>Local connectivity captures spatial structure</li>
            <li>Learns translation invariant features automatically</li>
            <li>Hierarchical feature learning through multiple layers</li>
            <li>Efficient computation through convolution operation</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
