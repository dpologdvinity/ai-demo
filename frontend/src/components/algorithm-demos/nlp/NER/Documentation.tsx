import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About NER</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is NER?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Named Entity Recognition (NER) is the task of identifying and classifying
            named entities in text such as persons, organizations, locations, dates,
            and other specific categories. It's a key NLP task for information extraction
            and understanding the semantic meaning of text.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Tokenize text into words and sentences</li>
            <li>Use trained neural networks or rule-based patterns</li>
            <li>Assign entity type labels to token sequences</li>
            <li>Optionally merge multi-token entities</li>
            <li>Return entity spans with confidence scores</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Information extraction from documents</li>
            <li>Knowledge graph construction</li>
            <li>Resume screening and recruiting</li>
            <li>News article analysis</li>
            <li>Entity-based search and recommendation</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
