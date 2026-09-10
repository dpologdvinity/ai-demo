import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About BERT Fine-tuning</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is BERT Fine-tuning?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model
            that has been trained on massive text corpora. Fine-tuning involves taking this pre-trained model
            and adapting it to a specific task, such as text classification, using a smaller labeled dataset.
            This transfer learning approach enables rapid model development with state-of-the-art performance
            while requiring significantly less training data than training from scratch.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Load pre-trained BERT model and tokenizer from Hugging Face</li>
            <li>Tokenize input texts with special tokens ([CLS], [SEP]) to max length</li>
            <li>Add classification layer on top of BERT's output representation</li>
            <li>Fine-tune the entire model with a small learning rate using labeled data</li>
            <li>Evaluate on validation set and track loss/accuracy per epoch</li>
            <li>Use trained model to make predictions on new texts</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Parameters
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Learning Rate:</strong> Controls gradient step size. Smaller rates (1e-5 to 5e-5) preserve
              pre-trained knowledge while adapting to new task.
            </li>
            <li>
              <strong>Epochs:</strong> Number of complete passes through training data. More epochs improve
              performance but risk overfitting on small datasets.
            </li>
            <li>
              <strong>Max Sequence Length:</strong> Maximum tokens per input text. Longer sequences preserve context
              but require more memory.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Common Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Sentiment analysis and opinion mining</li>
            <li>Intent detection in chatbots and voice assistants</li>
            <li>Named entity recognition and information extraction</li>
            <li>Question answering systems</li>
            <li>Text categorization and document classification</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>State-of-the-art performance with less training data</li>
            <li>Bidirectional context understanding from both directions</li>
            <li>Pre-trained on massive text corpora (Wikipedia, BookCorpus)</li>
            <li>Attention mechanism provides interpretable focus weights</li>
            <li>Transfer learning reduces training time significantly</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Limitations
          </h3>
          <ul className="list-disc list-inside space-y-1 text-red-700 dark:text-red-300">
            <li>High computational requirements (GPU recommended for reasonable speed)</li>
            <li>Large model size (110M+ parameters for BERT-base)</li>
            <li>Longer inference time compared to simpler models</li>
            <li>Risk of overfitting on very small datasets</li>
            <li>Hyperparameter sensitivity requiring careful tuning</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
