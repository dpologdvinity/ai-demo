import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Batch Normalization</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Batch Normalization?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Batch Normalization (BN) is a technique that normalizes the input to each layer by standardizing the activations
            from the previous layer. It reduces internal covariate shift, allowing networks to train faster and more stably
            with higher learning rates.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Calculate mean and variance of activations in the batch</li>
            <li>Normalize activations to zero mean and unit variance</li>
            <li>Apply learnable scale (gamma) and shift (beta) parameters</li>
            <li>Maintain running statistics for use during inference</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Benefits
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Accelerates training (typically 2-10x faster convergence)</li>
            <li>Allows higher learning rates without divergence</li>
            <li>Reduces sensitivity to weight initialization</li>
            <li>Acts as a regularizer, reducing need for dropout</li>
            <li>Enables training of deeper networks</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Parameters
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Momentum:</strong> Decay factor for running statistics (typically 0.1-0.9)
            </li>
            <li>
              <strong>Epsilon:</strong> Small constant for numerical stability (typically 1e-5)
            </li>
            <li>
              <strong>Affine:</strong> Whether to learn scale and shift parameters</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
