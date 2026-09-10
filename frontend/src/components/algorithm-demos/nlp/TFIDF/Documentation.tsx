import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About TF-IDF</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is TF-IDF?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical technique that reflects
            how important a word is to a document in a collection of documents. It assigns higher scores
            to terms that are frequent in specific documents but rare across the entire corpus, making
            it useful for identifying distinctive keywords.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>
              <strong>TF (Term Frequency):</strong> Count how often a term appears in a document
            </li>
            <li>
              <strong>IDF (Inverse Document Frequency):</strong> Measure how unique a term is across all documents
            </li>
            <li>Multiply TF × IDF to get the final importance score</li>
            <li>Optional: Normalize vectors to unit length for comparability</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Concepts
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>TF:</strong> Raw frequency or normalized by document length. High frequency in a
              document makes a term more important.
            </li>
            <li>
              <strong>IDF:</strong> Inverse of document frequency. Rare words get higher IDF scores,
              common words (the, a, is) get penalized.
            </li>
            <li>
              <strong>TF-IDF Score:</strong> Balances local relevance (TF) with global rarity (IDF).
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Simple, interpretable weighting scheme</li>
            <li>Effective for identifying important keywords</li>
            <li>Widely used in information retrieval and search</li>
            <li>Works well as baseline for document similarity</li>
            <li>Computational efficiency</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Limitations
          </h3>
          <ul className="list-disc list-inside space-y-1 text-red-700 dark:text-red-300">
            <li>Doesn't capture semantic relationships between terms</li>
            <li>Sensitive to stop words and preprocessing</li>
            <li>High-dimensional sparse vectors</li>
            <li>Assumes term independence</li>
            <li>Penalizes rare but important domain-specific terms</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Information retrieval and search ranking</li>
            <li>Document similarity and clustering</li>
            <li>Feature extraction for text classification</li>
            <li>Keyword extraction and summarization</li>
            <li>Topic analysis and exploration</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
