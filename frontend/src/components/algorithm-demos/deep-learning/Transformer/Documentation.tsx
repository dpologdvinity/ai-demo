import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Transformers</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is a Transformer?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            The Transformer is a neural network architecture based entirely on attention mechanisms, eliminating recurrence entirely. It processes entire sequences in parallel using self-attention to relate different positions. This architecture powers modern language models like GPT and BERT and has become the foundation of deep learning.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Components
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Self-Attention:</strong> Each position attends to all other positions
            </li>
            <li>
              <strong>Multi-Head Attention:</strong> Multiple parallel attention mechanisms
            </li>
            <li>
              <strong>Positional Encoding:</strong> Encodes sequence position information
            </li>
            <li>
              <strong>Feedforward Network:</strong> Two-layer MLP applied per position
            </li>
            <li>
              <strong>Layer Normalization:</strong> Stabilizes training
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Machine translation and NLP</li>
            <li>Language modeling and text generation</li>
            <li>Question answering systems</li>
            <li>Image recognition (Vision Transformers)</li>
            <li>Any sequence-to-sequence task</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
