import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: unknown;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About VGG</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is VGG?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            VGG (Visual Geometry Group Networks) is a deep convolutional neural network
            architecture that emphasizes simplicity and depth. It uses uniform 3x3 convolutional
            filters stacked in sequence within convolutional blocks, followed by max pooling.
            The straightforward design principle makes VGG easy to understand while achieving
            excellent accuracy on ImageNet classification. VGG was trained on a large-scale
            image dataset and serves as a popular baseline architecture and feature extractor.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Input image (224x224 pixels) passes through initial convolution layer</li>
            <li>Five convolutional blocks, each with stacked 3x3 convolutions</li>
            <li>Max pooling layers reduce spatial dimensions between blocks</li>
            <li>ReLU activation functions after each convolution</li>
            <li>Global average pooling or flattening reduces feature maps to vectors</li>
            <li>Three fully connected layers process the flattened features</li>
            <li>Softmax produces probability distribution over 1000 classes</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Variants
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>VGG-11:</strong> 11 layers, lightweight variant
            </li>
            <li>
              <strong>VGG-13:</strong> 13 layers, compact version
            </li>
            <li>
              <strong>VGG-16:</strong> 16 layers, most commonly used variant
            </li>
            <li>
              <strong>VGG-19:</strong> 19 layers, deepest variant with best accuracy
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Simple and straightforward architecture design</li>
            <li>Excellent interpretability - easy to understand each layer</li>
            <li>Strong baseline performance on image classification</li>
            <li>Pre-trained models widely available</li>
            <li>Effective feature extraction for transfer learning</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Limitations
          </h3>
          <ul className="list-disc list-inside space-y-1 text-red-700 dark:text-red-300">
            <li>Very large number of parameters (138M for VGG-16)</li>
            <li>Computationally expensive - slow inference compared to ResNet</li>
            <li>High memory requirements during training</li>
            <li>Not as deep as modern architectures like ResNet</li>
            <li>Slow convergence during training</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Image classification (ImageNet, custom datasets)</li>
            <li>Feature extraction for transfer learning</li>
            <li>Texture recognition</li>
            <li>Style transfer (popular for neural style transfer)</li>
            <li>Fine-grained image classification</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            VGG vs ResNet
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li><strong>Architecture:</strong> VGG uses sequential convolutions, ResNet uses skip connections</li>
            <li><strong>Depth:</strong> VGG typically 16-19 layers, ResNet can be 50-101+ layers</li>
            <li><strong>Parameters:</strong> VGG has many more parameters for similar depth</li>
            <li><strong>Speed:</strong> ResNet is significantly faster due to efficiency</li>
            <li><strong>Use:</strong> VGG for understanding CNNs, ResNet for production systems</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
