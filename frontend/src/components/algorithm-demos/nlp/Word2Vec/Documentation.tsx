import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Word2Vec</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Word2Vec?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Word2Vec is a neural network-based technique that learns word embeddings from large text
            corpora. It represents each word as a dense vector in a continuous vector space where words with
            similar meanings are positioned close to each other. This allows the model to capture semantic
            and syntactic relationships between words.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Two Training Algorithms
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>CBOW (Continuous Bag of Words):</strong> Predicts the target word given its context.
              Works well with small datasets but is slower to train. Better for frequent words.
            </li>
            <li>
              <strong>Skip-gram:</strong> Predicts context words given a target word. Faster to train and
              works better with large datasets. Better for rare words and semantic relationships.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Initialize random word vectors</li>
            <li>For each word in context window, predict target word (CBOW) or context (Skip-gram)</li>
            <li>Calculate prediction error and update vectors via backpropagation</li>
            <li>Repeat for multiple epochs until convergence</li>
            <li>Final vectors capture semantic relationships learned from the corpus</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Concepts
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Context Window:</strong> Number of words before/after the target word. Larger windows
              capture broader semantic relationships.
            </li>
            <li>
              <strong>Vector Dimension:</strong> Dimensionality of embedding vectors. Higher dimensions
              capture more information but require more data.
            </li>
            <li>
              <strong>Min Count:</strong> Minimum frequency threshold. Rare words are filtered out.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Fast to train compared to other deep learning methods</li>
            <li>Captures semantic and syntactic relationships effectively</li>
            <li>Produces dense, efficient representations</li>
            <li>Works well as input to other machine learning models</li>
            <li>Handles large vocabularies efficiently</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Limitations
          </h3>
          <ul className="list-disc list-inside space-y-1 text-red-700 dark:text-red-300">
            <li>Fixed embeddings - same word has same vector regardless of context</li>
            <li>Cannot handle out-of-vocabulary words</li>
            <li>Sensitive to hyperparameter choices (window, vector size, epochs)</li>
            <li>Requires large corpus for high-quality embeddings</li>
            <li>Training is non-deterministic even with fixed seed</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Feature extraction for text classification and clustering</li>
            <li>Word similarity and semantic search</li>
            <li>Machine translation and cross-lingual analysis</li>
            <li>Building recommendation systems based on text</li>
            <li>Named entity recognition and part-of-speech tagging</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Related Techniques
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>
              <strong>GloVe:</strong> Combines matrix factorization with local context window methods.
            </li>
            <li>
              <strong>FastText:</strong> Extension of Word2Vec that handles subword information.
            </li>
            <li>
              <strong>BERT, ELMo:</strong> Contextual embeddings for better context-dependent representations.
            </li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
