import React from 'react';
import { Card, CardContent } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  similar_words: Array<{ word: string; similarity: number }>;
  analogy_result?: {
    query: string;
    result_word: string;
    similarity: number;
    top_results: Array<{ word: string; similarity: number }>;
  };
  metrics: Record<string, any>;
  parameters_used: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  return (
    <div className="space-y-4">
      {/* Similar Words */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Similar Words
        </h4>
        <div className="space-y-2">
          {result.similar_words.slice(0, 10).map((item, idx) => (
            <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded">
              <span className="font-mono text-sm text-gray-700 dark:text-gray-300">
                {item.word}
              </span>
              <div className="flex items-center gap-2">
                <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full"
                    style={{ width: `${item.similarity * 100}%` }}
                  />
                </div>
                <span className="text-xs text-gray-600 dark:text-gray-400 w-12 text-right">
                  {(item.similarity * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Analogy Results */}
      {result.analogy_result && (
        <div className="border-t pt-4">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
            Analogy Result
          </h4>
          <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg mb-3">
            <p className="text-sm text-purple-900 dark:text-purple-100 mb-2">
              <strong>Query:</strong> {result.analogy_result.query}
            </p>
            <p className="text-xl font-bold text-purple-700 dark:text-purple-300">
              {result.analogy_result.result_word}
            </p>
            <p className="text-xs text-purple-800 dark:text-purple-200 mt-1">
              Confidence: {(result.analogy_result.similarity * 100).toFixed(1)}%
            </p>
          </div>

          <h5 className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-2 uppercase">
            Top Analogy Matches
          </h5>
          <div className="space-y-2">
            {result.analogy_result.top_results.slice(0, 5).map((item, idx) => (
              <div key={idx} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded">
                <span className="font-mono text-xs text-gray-700 dark:text-gray-300">
                  {item.word}
                </span>
                <span className="text-xs text-gray-600 dark:text-gray-400">
                  {(item.similarity * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Info */}
      <div className="grid grid-cols-2 gap-3">
        <Card>
          <CardContent className="pt-4">
            <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Vocabulary
            </p>
            <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {result.metrics.vocab_size || 0}
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">words</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Dimensions
            </p>
            <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
              {result.parameters_used.embedding_dim}
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">dimensions</p>
          </CardContent>
        </Card>
      </div>

      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> GloVe captures semantic relationships. Similar words
          have high similarity scores (close to 1.0). Word analogies reveal linguistic patterns, e.g.,
          king - man + woman ≈ queen.
        </p>
      </div>
    </div>
  );
}
