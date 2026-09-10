import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Semantic Segmentation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Semantic Segmentation?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Semantic segmentation assigns a class label to each pixel in an image, enabling
            detailed scene understanding. Unlike object detection which draws bounding boxes,
            semantic segmentation provides pixel-level accuracy. DeepLabV3 uses Atrous Spatial
            Pyramid Pooling (ASPP) to capture multi-scale context through dilated convolutions
            at multiple rates.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Backbone network extracts features at multiple scales</li>
            <li>ASPP module captures multi-scale context via dilated convolutions</li>
            <li>Decoder gradually recovers spatial information</li>
            <li>Final classification layer produces per-pixel class predictions</li>
            <li>Output is a dense segmentation mask covering entire image</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Technologies
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Atrous Convolution:</strong> Dilated filters increase receptive field
              without reducing resolution
            </li>
            <li>
              <strong>Multi-scale Processing:</strong> ASPP processes at multiple scales
            </li>
            <li>
              <strong>Skip Connections:</strong> Preserve spatial information during decoding
            </li>
            <li>
              <strong>Dense Prediction:</strong> Produces output at same resolution as input
            </li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
