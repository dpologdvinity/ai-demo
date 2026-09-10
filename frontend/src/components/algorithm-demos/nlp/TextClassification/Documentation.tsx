import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Text Classification</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Text Classification?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Text Classification is the task of assigning predefined categories or labels to
            text documents. It's one of the most common NLP tasks with applications ranging
            from sentiment analysis to topic categorization, spam detection, and intent
            classification for chatbots.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Extract features from text using TF-IDF vectorization</li>
            <li>Train a supervised classifier on labeled data</li>
            <li>Split data into training and test sets</li>
            <li>Evaluate model performance using various metrics</li>
            <li>Make predictions on new documents</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Sentiment analysis (positive/negative/neutral)</li>
            <li>Email spam detection</li>
            <li>News topic categorization</li>
            <li>Customer feedback classification</li>
            <li>Intent detection for conversational AI</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
