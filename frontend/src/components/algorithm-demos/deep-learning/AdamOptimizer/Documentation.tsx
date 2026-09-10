import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Adam Optimizer</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Adam?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Adam (Adaptive Moment Estimation) combines advantages of AdaGrad and RMSprop
            optimizers with momentum. It maintains exponentially decaying averages of both
            squared gradients (second moment) and gradients (first moment), enabling adaptive
            per-parameter learning rates.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Compute gradient g_t for current batch</li>
            <li>Update first moment: m_t = β₁·m_(t-1) + (1-β₁)·g_t</li>
            <li>Update second moment: v_t = β₂·v_(t-1) + (1-β₂)·g_t²</li>
            <li>Bias correction: m̂_t = m_t / (1-β₁^t), v̂_t = v_t / (1-β₂^t)</li>
            <li>Update parameters: θ = θ - α·m̂_t / (√v̂_t + ε)</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Parameters
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Learning Rate (α):</strong> Controls optimization step size.
            </li>
            <li>
              <strong>Beta1 (β₁):</strong> Momentum decay rate, typically 0.9.
            </li>
            <li>
              <strong>Beta2 (β₂):</strong> Second moment decay, typically 0.999.
            </li>
            <li>
              <strong>Epsilon (ε):</strong> Numerical stability term, typically 1e-8.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Adaptive per-parameter learning rates</li>
            <li>Fast convergence on most problems</li>
            <li>Handles sparse gradients well</li>
            <li>Robust to learning rate choice</li>
            <li>Good default hyperparameters</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Limitations
          </h3>
          <ul className="list-disc list-inside space-y-1 text-red-700 dark:text-red-300">
            <li>Slower than SGD on some problems</li>
            <li>May not generalize as well as SGD</li>
            <li>Higher memory usage for moment estimates</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
