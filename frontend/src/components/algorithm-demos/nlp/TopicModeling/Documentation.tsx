import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Topic Modeling</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Topic Modeling?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Topic Modeling is an unsupervised machine learning technique that discovers abstract
            themes or topics in a collection of documents. Latent Dirichlet Allocation (LDA) is
            a widely used probabilistic model that treats documents as mixtures of topics and
            topics as mixtures of words.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How LDA Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Assume each document contains multiple topics in varying proportions</li>
            <li>Assume each topic contains multiple words in varying proportions</li>
            <li>Use iterative algorithms (Gibbs sampling) to infer topic distributions</li>
            <li>Represent topics as probability distributions over vocabulary</li>
            <li>Calculate document-topic and topic-word associations</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Document clustering and organization</li>
            <li>Exploratory analysis of large text collections</li>
            <li>Content recommendation systems</li>
            <li>Trend analysis in social media or news</li>
            <li>Taxonomy building from unstructured text</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
