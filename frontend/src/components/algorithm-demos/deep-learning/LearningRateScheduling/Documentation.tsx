import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Learning Rate Scheduling</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Learning Rate Scheduling?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Learning rate scheduling adjusts the learning rate during training to improve
            convergence speed, stability, and final performance. Instead of using a constant
            learning rate, schedules can decay, cycle, or adapt based on training progress.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Common Schedules
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Step Decay:</strong> Multiply learning rate by γ every N epochs.
            </li>
            <li>
              <strong>Exponential Decay:</strong> Gradually decrease learning rate exponentially.
            </li>
            <li>
              <strong>Cosine Annealing:</strong> Smoothly decay learning rate following cosine curve.
            </li>
            <li>
              <strong>Reduce on Plateau:</strong> Decrease learning rate when validation loss plateaus.
            </li>
            <li>
              <strong>Cyclic LR:</strong> Cycle between base and max learning rates during training.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Benefits
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Faster convergence with larger initial learning rates</li>
            <li>Better final performance with smaller learning rates later</li>
            <li>Escape local minima with cyclic schedules</li>
            <li>Automatic adaptation with plateau-based schedules</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Parameters
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Initial Learning Rate:</strong> Starting learning rate.
            </li>
            <li>
              <strong>Decay Factor (γ):</strong> Multiplicative reduction for step decay.
            </li>
            <li>
              <strong>Step Size:</strong> Epochs between adjustments.
            </li>
            <li>
              <strong>Min/Max LR:</strong> Bounds for learning rate range.
            </li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
