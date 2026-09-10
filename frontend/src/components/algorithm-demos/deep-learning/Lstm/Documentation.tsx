import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About LSTMs</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is an LSTM?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Long Short-Term Memory (LSTM) is a type of recurrent neural network designed to learn long-term dependencies in sequential data. LSTMs use specialized gate mechanisms (input, forget, and output gates) to control information flow, allowing them to remember or forget information selectively over longer sequences than standard RNNs.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Maintain a cell state that acts as memory</li>
            <li>Forget gate decides what information to discard</li>
            <li>Input gate determines what new information to add</li>
            <li>Cell state is updated with old and new information</li>
            <li>Output gate decides what to output from cell state</li>
            <li>Hidden state flows to next timestep</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Time series prediction with long-term patterns</li>
            <li>Language modeling and machine translation</li>
            <li>Speech and handwriting recognition</li>
            <li>Sentiment analysis</li>
            <li>Any task requiring memory of past context</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
