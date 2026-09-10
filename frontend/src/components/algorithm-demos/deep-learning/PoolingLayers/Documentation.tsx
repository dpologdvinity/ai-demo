import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Pooling Layers</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What are Pooling Layers?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Pooling layers downsample feature maps by aggregating values in local windows.
            They reduce spatial dimensions while preserving important features, decreasing
            computation and memory while improving robustness to small spatial shifts.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Pooling Types
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Max Pooling:</strong> Takes maximum value in window. Best for preserving
              strong activations and edge information.
            </li>
            <li>
              <strong>Average Pooling:</strong> Takes mean of window values. Smoother and more
              information-preserving than max pooling.
            </li>
            <li>
              <strong>Global Pooling:</strong> Reduces entire feature map to single value.
              Useful for transitioning to fully connected layers.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Benefits
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Reduces computation and memory requirements</li>
            <li>Provides translation invariance to input shifts</li>
            <li>Helps prevent overfitting</li>
            <li>Reduces number of parameters to learn</li>
            <li>Improves model robustness</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Parameters
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Pool Size:</strong> Dimensions of pooling window (2×2 or 3×3 typical)
            </li>
            <li>
              <strong>Stride:</strong> Step size for pooling window (often equals pool size)
            </li>
            <li>
              <strong>Padding:</strong> Zeros added around input to maintain dimensions
            </li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
