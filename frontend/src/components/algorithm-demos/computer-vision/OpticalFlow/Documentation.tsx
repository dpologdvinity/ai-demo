import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Optical Flow</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Optical Flow?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Optical flow estimates the motion of objects between consecutive video frames by analyzing
            changes in pixel intensities. It produces a vector field where each vector represents the
            apparent velocity of pixels, essential for video analysis, stabilization, and tracking.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Build Gaussian image pyramid for coarse-to-fine estimation</li>
            <li>Compute flow at coarse scales first</li>
            <li>Refine flow progressively at finer scales</li>
            <li>Use window-based or region-based matching</li>
            <li>Produce dense (all pixels) or sparse (feature points) motion vectors</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Concepts
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Dense Flow:</strong> Farneback computes motion for every pixel
            </li>
            <li>
              <strong>Sparse Flow:</strong> Lucas-Kanade computes motion at feature points
            </li>
            <li>
              <strong>Pyramid Levels:</strong> Multiple scales to handle large motions
            </li>
            <li>
              <strong>Magnitude:</strong> Speed of motion at each point
            </li>
            <li>
              <strong>Direction:</strong> Angle of motion vector
            </li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
