import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About RNNs</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is an RNN?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            A Recurrent Neural Network (RNN) is designed to process sequential data by maintaining a hidden state that captures information from previous time steps. Unlike feedforward networks, RNNs have connections that cycle back, allowing them to remember patterns in sequences of arbitrary length.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Process one element of the sequence at a time</li>
            <li>Maintain hidden state from previous timestep</li>
            <li>Combine current input with hidden state</li>
            <li>Apply activation function to compute new hidden state</li>
            <li>Generate output from hidden state</li>
            <li>Propagate hidden state to next timestep</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Time series prediction and forecasting</li>
            <li>Natural language processing (language models, translation)</li>
            <li>Speech recognition and synthesis</li>
            <li>Sequence-to-sequence tasks</li>
            <li>Any task involving sequential dependencies</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
