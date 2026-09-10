import React from 'react';
import { Card, CardContent } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  similar_words: Record<string, Array<{ word: string; similarity: number }>>;
  analogies: Array<{ query: string; result: string; similarity: number }>;
  metrics: Record<string, any>;
  parameters_used: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const sampleWords = Object.keys(result.similar_words).slice(0, 3);

  return (
    <div className="space-y-4">
      {/* Similar Words for Multiple Words */}
      {sampleWords.map((word) => (
        <div key={word}>
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Words similar to "{word}"
          </h4>
          <div className="space-y-2">
            {result.similar_words[word]?.slice(0, 8).map((item, idx) => (
              <div key={idx} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded">
                <span className="font-mono text-sm text-gray-700 dark:text-gray-300">
                  {item.word}
                </span>
                <div className="flex items-center gap-2">
                  <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                    <div
                      className="bg-green-500 h-1.5 rounded-full"
                      style={{ width: `${item.similarity * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400 w-10 text-right">
                    {(item.similarity * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}

      {/* Word Analogies */}
      {result.analogies.length > 0 && (
        <div className="border-t pt-4">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
            Word Analogies
          </h4>
          <div className="space-y-2">
            {result.analogies.slice(0, 5).map((item, idx) => (
              <div key={idx} className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">
                  {item.query}
                </p>
                <p className="font-mono font-bold text-purple-700 dark:text-purple-300">
                  → {item.result}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  Confidence: {(item.similarity * 100).toFixed(1)}%
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Statistics */}
      <div className="grid grid-cols-3 gap-3">
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
              {result.parameters_used.vector_size}
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">D</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Algorithm
            </p>
            <p className="text-lg font-bold text-green-600 dark:text-green-400">
              {result.parameters_used.sg === 0 ? 'CBOW' : 'Skip-g'}
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">method</p>
          </CardContent>
        </Card>
      </div>

      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> Word2Vec learns dense vector representations where semantically
          similar words have close vectors. Higher similarity scores (close to 1.0) indicate strong semantic relationships.
        </p>
      </div>
    </div>
  );
}
