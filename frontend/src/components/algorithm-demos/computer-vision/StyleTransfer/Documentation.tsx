import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Style Transfer</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Neural Style Transfer?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Neural Style Transfer combines the content of one image with the artistic style of
            another using a pre-trained CNN. Introduced by Gatys et al. in 2015, it uses VGG19
            to extract feature representations at multiple layers. Content is captured by
            high-level features, while style is represented by Gram matrices that capture
            texture and artistic patterns.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Start with a random or content-initialized image</li>
            <li>Extract feature representations using pre-trained VGG19 network</li>
            <li>Compute content loss as MSE between content features</li>
            <li>Compute style loss as MSE between Gram matrices</li>
            <li>Iteratively optimize image to minimize weighted combination of losses</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Concepts
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Gram Matrix:</strong> Captures style by computing correlations between
              feature maps
            </li>
            <li>
              <strong>Content Loss:</strong> MSE between content features from conv4_2 layer
            </li>
            <li>
              <strong>Style Loss:</strong> MSE between Gram matrices from multiple layers
            </li>
            <li>
              <strong>Weight Balancing:</strong> Content/style weights control the balance
              between preservation and stylization
            </li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
