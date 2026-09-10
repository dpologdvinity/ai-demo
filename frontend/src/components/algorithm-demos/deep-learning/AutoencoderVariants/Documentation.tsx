import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Autoencoder Variants</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What are Autoencoder Variants?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Autoencoder variants extend the basic autoencoder architecture with modifications to address
            specific limitations or learn different types of representations. Each variant adds different
            constraints or objectives to improve performance on specific tasks.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Types of Variants
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Vanilla Autoencoder:</strong> The basic autoencoder with standard MSE reconstruction loss.
              Simple and effective for dimensionality reduction.
            </li>
            <li>
              <strong>Denoising Autoencoder:</strong> Trains to reconstruct clean data from noisy input.
              Forces the encoder to learn robust features invariant to noise.
            </li>
            <li>
              <strong>Sparse Autoencoder:</strong> Uses L1 regularization to encourage sparsity in the latent
              representation. Learns a sparse set of features per sample.
            </li>
            <li>
              <strong>Contractive Autoencoder:</strong> Penalizes sensitivity to input perturbations,
              making the representation more robust to small changes in the input.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Denoising: Data cleaning and noise removal</li>
            <li>Feature learning: Discovering sparse, interpretable features</li>
            <li>Robustness: Learning invariant representations</li>
            <li>Anomaly detection: Using reconstruction error as anomaly score</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
