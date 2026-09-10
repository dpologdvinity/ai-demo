import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About POS Tagging</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is POS Tagging?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Part-of-Speech (POS) tagging is the process of assigning grammatical categories
            (nouns, verbs, adjectives, etc.) to each word in a sentence. It's a fundamental
            NLP task that serves as a building block for many downstream applications like
            parsing, named entity recognition, and machine translation.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Tokenize text into individual words</li>
            <li>Use statistical or rule-based models to predict grammatical category</li>
            <li>Consider context and word patterns</li>
            <li>Optionally parse grammatical dependencies</li>
            <li>Return tagged output with confidence scores</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Natural language understanding and analysis</li>
            <li>Information extraction and named entity recognition</li>
            <li>Machine translation and language generation</li>
            <li>Grammar checking and text analysis</li>
            <li>Building blocks for dependency parsing</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
