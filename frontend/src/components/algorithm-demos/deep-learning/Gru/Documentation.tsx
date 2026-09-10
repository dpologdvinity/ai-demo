import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About GRUs</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is a GRU?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            A Gated Recurrent Unit (GRU) is a simplified variant of the LSTM that combines the forget and input gates into a single update gate. It has fewer parameters than LSTM while maintaining similar performance, making it computationally more efficient for many applications.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Reset gate determines how much past information to ignore</li>
            <li>Update gate decides how much new vs old hidden state to keep</li>
            <li>Compute candidate hidden state using reset gate</li>
            <li>Blend old and new hidden states using update gate</li>
            <li>Pass combined hidden state to next timestep</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Time series forecasting with fewer parameters</li>
            <li>Language modeling and NLP tasks</li>
            <li>Speech and music generation</li>
            <li>Real-time applications requiring efficiency</li>
            <li>Tasks where computational speed is critical</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
