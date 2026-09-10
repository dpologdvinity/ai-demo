import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About SIFT</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is SIFT?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            SIFT (Scale-Invariant Feature Transform) detects and describes distinctive local features
            in images that are invariant to scale, rotation, and illumination changes. Each feature is
            represented by a 128-dimensional descriptor, enabling reliable matching between images.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Build Gaussian pyramid and compute Difference-of-Gaussians</li>
            <li>Detect local extrema across scales and space</li>
            <li>Localize keypoints and eliminate weak features</li>
            <li>Assign dominant orientation to each keypoint</li>
            <li>Compute 128-dimensional feature descriptors</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Concepts
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Octaves:</strong> Image pyramid levels; more octaves capture larger features
            </li>
            <li>
              <strong>Scale:</strong> Size at which feature was detected (invariant to scale change)
            </li>
            <li>
              <strong>Response:</strong> Strength/confidence of the keypoint detection
            </li>
            <li>
              <strong>Descriptor:</strong> 128D vector encoding local gradient information
            </li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
