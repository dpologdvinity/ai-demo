import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: unknown;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About ResNet</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is ResNet?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            ResNet (Residual Networks) is a deep convolutional neural network architecture
            that introduced residual connections (skip connections) to enable training of
            very deep networks. The key innovation is the residual block, which learns the
            residual (difference) between input and output, rather than learning the output
            directly. This enables much deeper architectures (up to 1000 layers) while
            maintaining training stability.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Input image is passed through initial convolution and pooling layer</li>
            <li>Series of residual blocks process the image, each learning residual features</li>
            <li>Skip connections bypass 1-2 convolutional layers, allowing direct gradient flow</li>
            <li>Global average pooling reduces spatial dimensions to 1x1</li>
            <li>Fully connected layer produces 1000 class predictions (ImageNet classes)</li>
            <li>Softmax produces probability distribution over classes</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Variants
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>ResNet-18:</strong> 18 layers, lightweight, fast inference
            </li>
            <li>
              <strong>ResNet-34:</strong> 34 layers, moderate depth and compute
            </li>
            <li>
              <strong>ResNet-50:</strong> 50 layers with bottleneck blocks, better accuracy
            </li>
            <li>
              <strong>ResNet-101:</strong> 101 layers, deeper variant for high accuracy
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Enables training of very deep networks (100+ layers)</li>
            <li>State-of-the-art accuracy on ImageNet classification</li>
            <li>Skip connections solve vanishing gradient problem</li>
            <li>Highly flexible - can be combined with other techniques</li>
            <li>Well-researched with extensive pre-trained models available</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Image classification (ImageNet, CIFAR, custom datasets)</li>
            <li>Object detection (as backbone for Faster R-CNN, YOLO, etc.)</li>
            <li>Semantic segmentation</li>
            <li>Face recognition</li>
            <li>Medical image analysis</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Training Tips
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Use batch normalization to stabilize training</li>
            <li>Pre-trained ImageNet weights provide excellent starting point</li>
            <li>Fine-tune on custom datasets for better performance</li>
            <li>Use learning rate scheduling for improved convergence</li>
            <li>Data augmentation helps prevent overfitting on small datasets</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
