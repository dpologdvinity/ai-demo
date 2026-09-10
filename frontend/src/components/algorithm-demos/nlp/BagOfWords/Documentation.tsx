import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Bag of Words</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Bag of Words?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Bag of Words (BoW) is a simple text representation technique that converts documents into
            vectors based on word frequencies. Each document is represented as a vector where each
            position corresponds to a word in the vocabulary, and the value is the frequency of that
            word in the document. Word order and grammar are ignored.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Build a vocabulary from all unique words in the corpus</li>
            <li>For each document, count the frequency of each word</li>
            <li>Represent each document as a vector with word frequencies</li>
            <li>Optional: Apply term weighting schemes or binary encoding</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Parameters
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Max Features:</strong> Limits vocabulary size to the most frequent words,
              reducing dimensionality.
            </li>
            <li>
              <strong>N-gram Range:</strong> Captures word sequences (unigrams, bigrams, or both).
            </li>
            <li>
              <strong>Min/Max Document Frequency:</strong> Filters common and rare words.
            </li>
            <li>
              <strong>Binary:</strong> Use presence/absence instead of raw frequencies.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Simple and easy to understand</li>
            <li>Computationally efficient</li>
            <li>Good baseline for text classification</li>
            <li>Works well with sparse data</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Limitations
          </h3>
          <ul className="list-disc list-inside space-y-1 text-red-700 dark:text-red-300">
            <li>Ignores word order and grammar</li>
            <li>High-dimensional sparse vectors</li>
            <li>Doesn't capture semantic meaning</li>
            <li>Common words dominate the representation</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Document classification and spam detection</li>
            <li>Text similarity and clustering</li>
            <li>Baseline feature extraction for NLP tasks</li>
            <li>Keyword extraction and analysis</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
