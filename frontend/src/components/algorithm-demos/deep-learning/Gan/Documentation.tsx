import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About GANs</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is a GAN?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            A Generative Adversarial Network (GAN) consists of two neural networks competing in a game:
            the Generator (G) learns to create realistic fake samples from random noise, while the
            Discriminator (D) learns to distinguish real samples from generator-created fakes. This
            adversarial dynamic forces both networks to improve iteratively.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            The Adversarial Process
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Generator creates fake samples from random noise vectors</li>
            <li>Discriminator evaluates both real data and generator samples</li>
            <li>Discriminator loss: correctly classify real as real and fake as fake</li>
            <li>Generator loss: fool discriminator into classifying fakes as real</li>
            <li>Both networks update weights via alternating gradient descent</li>
            <li>Training reaches equilibrium when discriminator cannot improve</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Characteristics
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>No explicit likelihood:</strong> GANs don't model explicit probability distributions,
              learning through implicit adversarial training.
            </li>
            <li>
              <strong>Mode collapse:</strong> Generator may converge to producing limited variety of samples.
            </li>
            <li>
              <strong>Unstable training:</strong> Balancing G and D losses is challenging; hyperparameter
              tuning is critical.
            </li>
            <li>
              <strong>Powerful generation:</strong> Can generate highly realistic samples when trained properly.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Common Applications
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Image synthesis and style transfer</li>
            <li>Data augmentation for training datasets</li>
            <li>Super-resolution and image enhancement</li>
            <li>Domain adaptation and transfer learning</li>
            <li>Video frame prediction and generation</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
