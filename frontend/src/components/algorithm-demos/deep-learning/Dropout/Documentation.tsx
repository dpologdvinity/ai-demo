import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Dropout</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Dropout?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Dropout is a regularization technique that randomly deactivates neurons during training
            with probability p (dropout rate). This forces the network to learn redundant representations
            and prevents co-adaptation of neurons, effectively reducing overfitting.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>During training: Randomly drop each neuron with probability p</li>
            <li>Remaining neurons: Scale activations by 1/(1-p) to maintain expected values</li>
            <li>During inference: Use all neurons without any dropping</li>
            <li>Effect: Creates an ensemble of sub-networks sharing weights</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Properties
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Dropout rate p: Higher values (e.g., 0.5) drop more neurons</li>
            <li>Usually applied to hidden layers, rarely to input or output</li>
            <li>Training and inference modes differ (inverted dropout)</li>
            <li>Computational cost: Minimal during training and zero during inference</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Benefits
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Reduces overfitting significantly</li>
            <li>Improves generalization to unseen data</li>
            <li>Acts as model ensemble during training</li>
            <li>Simple to implement with minimal computational overhead</li>
            <li>Works well with large networks</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Best Practices
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Typical rates: 0.2-0.5 (20-50% drop rate)</li>
            <li>Higher rates for larger networks</li>
            <li>Apply to hidden layers, not input or output</li>
            <li>Disable during evaluation/inference</li>
            <li>Combine with other regularization (L1/L2, batch norm)</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
