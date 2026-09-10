import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Activation Functions</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What Are Activation Functions?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Activation functions introduce non-linearity into neural networks, enabling them to learn
            complex patterns. Without activation functions, stacking layers would be equivalent to a
            single linear transformation.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Common Functions
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>ReLU:</strong> f(x) = max(0, x). Fast, widely used, but suffers from dead neurons problem.
            </li>
            <li>
              <strong>Leaky ReLU:</strong> f(x) = max(αx, x). Addresses dead neurons by allowing small negative values.
            </li>
            <li>
              <strong>Sigmoid:</strong> f(x) = 1/(1 + e^(-x)). Outputs between 0 and 1, useful for binary classification.
            </li>
            <li>
              <strong>Tanh:</strong> f(x) = (e^x - e^(-x))/(e^x + e^(-x)). Zero-centered, often better than sigmoid.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Properties
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Range: Output bounds (e.g., sigmoid: [0,1], tanh: [-1,1])</li>
            <li>Smoothness: Continuous derivatives enable backpropagation</li>
            <li>Computational efficiency: Training speed impact</li>
            <li>Vanishing gradient: Sigmoid and tanh suffer with deep networks</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Dead Neurons Problem
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            ReLU activations output zero for all negative inputs. If a neuron enters this regime
            during training, its gradient becomes zero and it never updates again—becoming permanently
            "dead". Leaky ReLU fixes this by allowing a small negative slope.
          </p>
        </section>
      </CardContent>
    </Card>
  );
}
