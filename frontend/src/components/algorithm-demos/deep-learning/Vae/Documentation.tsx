import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About VAE</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is a Variational Autoencoder?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            A Variational Autoencoder (VAE) is a generative model that learns a continuous latent space
            distribution from data. Unlike standard autoencoders that learn discrete representations, VAEs
            learn to map data to a continuous probability distribution, enabling both reconstruction and
            generation of new samples. The key innovation is regularizing the latent space to follow a
            standard normal distribution using the KL divergence.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Encoder maps input to mean and variance of a latent distribution</li>
            <li>Sample from latent distribution using reparameterization trick</li>
            <li>Decoder reconstructs input from latent sample</li>
            <li>Loss combines reconstruction error and KL divergence from standard normal</li>
            <li>Training optimizes both reconstruction quality and latent space regularity</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Differences from Autoencoder
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>VAE learns a continuous distribution, not discrete codes</li>
            <li>Can generate new samples by sampling from latent distribution</li>
            <li>Latent space is regularized to be smooth and continuous</li>
            <li>Supports interpolation between samples in latent space</li>
            <li>Trade-off between reconstruction quality and latent space smoothness (beta parameter)</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Data generation and synthesis</li>
            <li>Latent space interpolation for smooth transitions</li>
            <li>Dimensionality reduction with probabilistic interpretation</li>
            <li>Semi-supervised learning</li>
            <li>Anomaly detection using reconstruction probability</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
