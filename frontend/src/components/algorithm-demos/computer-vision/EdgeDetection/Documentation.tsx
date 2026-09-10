import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Edge Detection</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Edge Detection?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Edge detection is a fundamental image processing technique that identifies boundaries
            between regions of different intensities. The Canny edge detector is a multi-stage algorithm
            that detects edges by finding local maxima of the image gradient, providing high-quality
            edge maps with single-pixel precision.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Gaussian blur to reduce noise</li>
            <li>Compute intensity gradients using Sobel operators</li>
            <li>Non-maximum suppression to thin edges</li>
            <li>Double thresholding to classify edge pixels</li>
            <li>Edge tracking by hysteresis to connect edges</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Concepts
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Threshold1:</strong> Lower threshold - edges weaker than this are discarded
            </li>
            <li>
              <strong>Threshold2:</strong> Upper threshold - edges stronger than this are definitely kept
            </li>
            <li>
              <strong>Hysteresis:</strong> Weak edges are kept only if connected to strong edges
            </li>
            <li>
              <strong>Aperture Size:</strong> Sobel kernel size affects edge smoothness
            </li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
