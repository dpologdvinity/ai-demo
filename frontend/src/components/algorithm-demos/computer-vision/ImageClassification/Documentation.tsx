import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Image Classification</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Image Classification?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Image classification is a computer vision task that assigns one or more labels to an
            image. Convolutional Neural Networks (CNNs) learn hierarchical features automatically
            from data, enabling them to classify images into predefined categories. Modern
            architectures like ResNet use residual connections to train very deep networks
            efficiently. Transfer learning with pre-trained models enables high accuracy without
            requiring massive training datasets.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Input image is resized to 224x224 pixels and normalized</li>
            <li>Convolutional layers extract hierarchical features at multiple scales</li>
            <li>Pooling layers reduce spatial dimensions while preserving important features</li>
            <li>Fully connected layers aggregate features and produce class predictions</li>
            <li>Softmax activation converts raw scores into probability distribution</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Concepts
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Convolution:</strong> Learned filters detect features like edges, textures,
              and patterns
            </li>
            <li>
              <strong>Residual Connections:</strong> Skip connections enable training deeper
              networks
            </li>
            <li>
              <strong>Transfer Learning:</strong> Pre-trained weights from ImageNet provide good
              starting features
            </li>
            <li>
              <strong>Softmax:</strong> Final activation converts logits to probabilities summing
              to 1.0
            </li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
