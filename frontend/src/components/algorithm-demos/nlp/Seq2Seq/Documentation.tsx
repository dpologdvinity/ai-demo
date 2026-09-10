import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Seq2Seq</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Seq2Seq?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Sequence-to-Sequence (Seq2Seq) is an encoder-decoder neural network architecture designed
            for transforming one sequence into another. The model consists of two main components:
            an encoder that processes the input sequence and creates a context representation, and
            a decoder that generates the output sequence from this context. Both typically use LSTM
            or GRU cells to handle sequential data effectively.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Encoder processes input sequence token by token, updating hidden state</li>
            <li>Final encoder hidden state becomes context vector for entire input</li>
            <li>Decoder uses context vector to generate output sequence token by token</li>
            <li>At each step, decoder predicts next token based on previous tokens and attention</li>
            <li>Attention mechanism computes weighted sum of encoder hidden states</li>
            <li>Teacher forcing during training uses ground truth as decoder input</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Parameters
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Hidden Size:</strong> Number of units in LSTM layers. Larger dimensions capture
              more complex patterns but require more computation.
            </li>
            <li>
              <strong>Attention:</strong> Allows decoder to focus on different input positions for each
              output token. Dramatically improves performance on long sequences.
            </li>
            <li>
              <strong>Teacher Forcing Ratio:</strong> Probability of using ground truth output during
              training (vs. model predictions). Higher values stabilize training.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Common Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Machine translation (e.g., English to French)</li>
            <li>Text summarization and paraphrasing</li>
            <li>Question answering systems</li>
            <li>Chatbots and conversational AI</li>
            <li>Speech recognition and text-to-speech</li>
            <li>Code generation and program synthesis</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Handles variable-length input and output sequences naturally</li>
            <li>Attention mechanism provides interpretable alignment information</li>
            <li>End-to-end differentiable - no feature engineering needed</li>
            <li>Single unified architecture for various sequence tasks</li>
            <li>Can capture long-range dependencies with attention</li>
            <li>Flexible architecture adaptable to different domains and modalities</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Limitations
          </h3>
          <ul className="list-disc list-inside space-y-1 text-red-700 dark:text-red-300">
            <li>Requires large amounts of parallel training data for good performance</li>
            <li>High computational cost to train (GPU recommended)</li>
            <li>Exposure bias: training uses ground truth but inference uses predictions</li>
            <li>May struggle with very long sequences even with attention</li>
            <li>Requires careful hyperparameter tuning</li>
            <li>Beam search decoding adds complexity at inference time</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
