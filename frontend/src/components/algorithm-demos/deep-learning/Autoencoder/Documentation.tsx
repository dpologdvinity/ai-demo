import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Autoencoders</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is an Autoencoder?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            An autoencoder is a neural network architecture designed for unsupervised learning that learns
            to compress data into a lower-dimensional representation (latent space) and then reconstruct it.
            The network consists of an encoder that compresses input data and a decoder that reconstructs
            the original data from the compressed representation. By forcing information through a bottleneck,
            autoencoders learn meaningful features without requiring labeled data.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Input data passes through encoder layers that progressively reduce dimensionality</li>
            <li>The encoder outputs a latent representation (compressed bottleneck)</li>
            <li>The decoder reconstructs the original data from the latent representation</li>
            <li>Network minimizes reconstruction loss (typically MSE) between input and output</li>
            <li>Learned latent space captures essential features of the data</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Data compression and dimensionality reduction</li>
            <li>Denoising data by removing noise during reconstruction</li>
            <li>Anomaly detection using reconstruction error as an anomaly score</li>
            <li>Feature extraction and unsupervised representation learning</li>
            <li>Image generation and interpolation in latent space</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
