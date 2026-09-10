import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About GloVe</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is GloVe?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm that
            learns word embeddings by combining the benefits of matrix factorization and local context
            window methods. It represents each word as a dense vector that captures semantic and syntactic
            relationships with other words.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Build a co-occurrence matrix of words in the corpus</li>
            <li>Define a weighted least squares objective to factorize this matrix</li>
            <li>Learn word vectors that minimize this objective via gradient descent</li>
            <li>The learned vectors encode semantic relationships between words</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Concepts
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Co-occurrence Matrix:</strong> Counts how often words appear together in context.
            </li>
            <li>
              <strong>Word Vector:</strong> Dense representation capturing semantic meaning and relationships.
            </li>
            <li>
              <strong>Cosine Similarity:</strong> Measures similarity between word vectors (0-1 scale).
            </li>
            <li>
              <strong>Word Analogies:</strong> Algebraic relationships between words (king - man + woman = queen).
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Captures both global and local word statistics</li>
            <li>Efficient to train compared to neural network methods</li>
            <li>Produces interpretable word relationships</li>
            <li>Performs well on word analogy and similarity tasks</li>
            <li>Scalable to large corpora</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Limitations
          </h3>
          <ul className="list-disc list-inside space-y-1 text-red-700 dark:text-red-300">
            <li>Requires large corpus for good quality embeddings</li>
            <li>Cannot handle out-of-vocabulary words</li>
            <li>Fixed vectors don't account for context (unlike contextual models)</li>
            <li>Sensitive to hyperparameters (window size, vector dimension)</li>
            <li>Training can be memory-intensive for large vocabularies</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Word similarity and semantic relatedness assessment</li>
            <li>Feature extraction for downstream NLP tasks</li>
            <li>Semantic search and document similarity</li>
            <li>Knowledge extraction and relationship discovery</li>
            <li>Building recommendation systems based on word relationships</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Modern Alternatives
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Word2Vec:</strong> Faster training with skip-gram and CBOW methods.
            </li>
            <li>
              <strong>FastText:</strong> Handles subword information and OOV words.
            </li>
            <li>
              <strong>BERT, ELMo:</strong> Contextual embeddings for better semantic understanding.
            </li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
