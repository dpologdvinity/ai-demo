import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Gradient Descent</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Gradient Descent?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Gradient descent is the fundamental optimization algorithm for training neural networks.
            It iteratively updates parameters in the direction opposite to the gradient of the loss
            function, moving toward a local minimum.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Algorithm Variants
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>SGD:</strong> Basic gradient descent with constant learning rate. Simple
              but can be slow and unstable.
            </li>
            <li>
              <strong>Momentum:</strong> Accumulates gradient directions, accelerating convergence
              and reducing oscillations.
            </li>
            <li>
              <strong>RMSprop:</strong> Adapts learning rate based on squared gradient magnitudes,
              handling different scales of parameters.
            </li>
            <li>
              <strong>Adam:</strong> Combines momentum and RMSprop for robust adaptive learning rates.
            </li>
            <li>
              <strong>AdaGrad:</strong> Accumulates squared gradients, automatically decreasing
              learning rate over time.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Concepts
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Gradient: Direction and magnitude of steepest ascent</li>
            <li>Learning Rate: Controls step size along gradient</li>
            <li>Convergence: When updates become negligibly small</li>
            <li>Local Minimum: Local optimum where gradient is near zero</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Challenges
          </h3>
          <ul className="list-disc list-inside space-y-1 text-red-700 dark:text-red-300">
            <li>Choosing appropriate learning rate</li>
            <li>Getting stuck in local minima</li>
            <li>Oscillations near minima (especially with high learning rates)</li>
            <li>Slow convergence in some regions</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
