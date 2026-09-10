import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Tokenization</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Tokenization?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Tokenization is the process of splitting text into smaller meaningful units called tokens.
            These tokens can be words, subwords, characters, or sentences depending on the tokenization
            strategy. It's a fundamental preprocessing step in natural language processing.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Tokenization Strategies
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Word:</strong> Splits text by whitespace and punctuation into individual words.
            </li>
            <li>
              <strong>Whitespace:</strong> Splits only on whitespace, preserving punctuation.
            </li>
            <li>
              <strong>Sentence:</strong> Splits text into sentences at periods, question marks, etc.
            </li>
            <li>
              <strong>WordPiece:</strong> Subword tokenization used in BERT, handles OOV words.
            </li>
            <li>
              <strong>BPE:</strong> Byte-Pair Encoding learns frequent character pairs.
            </li>
            <li>
              <strong>Character:</strong> Splits into individual characters.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Options
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Lowercase:</strong> Normalizes text case for case-insensitive analysis.
            </li>
            <li>
              <strong>Remove Punctuation:</strong> Cleans tokens by removing punctuation marks.
            </li>
            <li>
              <strong>Remove Stopwords:</strong> Filters common words (the, a, is) that add little meaning.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Importance
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Essential preprocessing for NLP pipelines</li>
            <li>Affects quality of downstream tasks</li>
            <li>Different strategies suit different applications</li>
            <li>Impacts vocabulary size and model performance</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Preprocessing for text classification</li>
            <li>Preparing text for embeddings and language models</li>
            <li>Information retrieval and search</li>
            <li>Text analysis and keyword extraction</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
