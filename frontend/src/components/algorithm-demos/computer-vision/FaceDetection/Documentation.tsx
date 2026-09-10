import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Face Detection</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Face Detection?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Face detection is a computer vision technique that locates faces in images. Haar Cascades use
            machine learning with Haar-like features to achieve fast, real-time face detection. The
            algorithm works by sliding detection windows across an image at multiple scales.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Compute Haar-like features that capture edge and line patterns</li>
            <li>Use AdaBoost to select discriminative features</li>
            <li>Cascade classifiers for fast rejection of non-face regions</li>
            <li>Slide detection window at multiple image scales</li>
            <li>Return bounding boxes for detected faces</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Parameters
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Scale Factor:</strong> Image scale reduction at each level (lower = more accurate)
            </li>
            <li>
              <strong>Min Neighbors:</strong> Minimum detections to confirm a face (higher = fewer false
              positives)
            </li>
            <li>
              <strong>Min/Max Size:</strong> Constrains face size search range
            </li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
