import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Data Augmentation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Data Augmentation?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Data augmentation is a technique that artificially expands the training dataset
            by applying random transformations to existing images. These transformations
            (rotations, flips, brightness changes, etc.) create new training samples that
            help models generalize better to unseen data.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Common Techniques
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Geometric:</strong> Rotations, flips, crops, scaling
            </li>
            <li>
              <strong>Color-based:</strong> Brightness, contrast, hue adjustments
            </li>
            <li>
              <strong>Noise:</strong> Adding random noise or blur
            </li>
            <li>
              <strong>Advanced:</strong> Cutmix, Mixup, AutoAugment
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Benefits
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Increases effective training set size without collecting new data</li>
            <li>Improves model generalization and robustness</li>
            <li>Reduces overfitting on small datasets</li>
            <li>Makes models invariant to transformations</li>
            <li>Simulates real-world data variations</li>
            <li>Improves model performance with limited data</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            When to Use
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Small datasets with limited training samples</li>
            <li>Tasks where transformations don't change class (e.g., rotation in digit recognition)</li>
            <li>Limited budget for data collection</li>
            <li>Need for model robustness to variations</li>
            <li>Imbalanced datasets (augment minority classes)</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
