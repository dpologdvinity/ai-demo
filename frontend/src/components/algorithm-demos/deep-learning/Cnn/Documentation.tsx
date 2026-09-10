import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About CNNs</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is a CNN?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Convolutional Neural Networks (CNNs) are deep learning architectures designed for
            processing grid-like data, especially images. They use convolutional layers that
            apply learned filters to extract spatial features, followed by pooling layers that
            reduce dimensions and fully connected layers for classification.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Components
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li><strong>Convolutional Layers:</strong> Extract features using learned filters</li>
            <li><strong>Pooling Layers:</strong> Reduce spatial dimensions via max or average pooling</li>
            <li><strong>Batch Normalization:</strong> Normalize activations for stable training</li>
            <li><strong>Activation Functions:</strong> Introduce non-linearity (ReLU, etc.)</li>
            <li><strong>Fully Connected Layers:</strong> Combine features for final classification</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How CNNs Work
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            CNNs learn hierarchical feature representations. Early layers detect low-level features
            (edges, textures), middle layers combine these into mid-level features (shapes, patterns),
            and deeper layers learn high-level semantic features (objects, concepts). This hierarchical
            learning makes CNNs highly effective for image understanding tasks.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Automatic feature extraction without manual engineering</li>
            <li>Translation invariance through weight sharing and pooling</li>
            <li>Hierarchical learning of increasingly complex features</li>
            <li>Significantly fewer parameters than fully connected networks</li>
            <li>State-of-the-art performance on image classification tasks</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Common Applications
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Image classification and object recognition</li>
            <li>Medical image analysis (radiology, pathology)</li>
            <li>Face recognition and verification</li>
            <li>Semantic segmentation</li>
            <li>Video analysis and action recognition</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
