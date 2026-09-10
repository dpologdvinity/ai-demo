import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Feedforward Networks</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is a Feedforward Network?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            A feedforward neural network (also called a multilayer perceptron) is a basic artificial neural network architecture where information flows in one direction—from input through hidden layers to output. Each neuron in one layer is connected to every neuron in the next layer, with no connections between neurons in the same layer or backward connections.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Input features are passed through the network</li>
            <li>Each hidden layer applies weighted transformations and activation functions</li>
            <li>Output layer produces predictions (classification probabilities or regression values)</li>
            <li>Loss is computed by comparing predictions to ground truth</li>
            <li>Backpropagation updates weights to minimize loss</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Classification tasks (deciding between predefined classes)</li>
            <li>Regression (predicting continuous values)</li>
            <li>Pattern recognition and feature learning</li>
            <li>Function approximation</li>
            <li>Building blocks for more complex architectures (CNNs, RNNs)</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
