import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface Topic {
  topic_id: number;
  top_words: Array<{ word: string; weight: number }>;
  keywords: string;
}

interface TrainingResult {
  success: boolean;
  topics: Topic[];
  coherence_score?: number;
  perplexity?: number;
  execution_time_ms: number;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  return (
    <div className="space-y-6">
      {/* Metrics Summary */}
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400">Topics Discovered</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {result.topics.length}
              </p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400">Coherence Score</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {result.coherence_score !== undefined
                  ? result.coherence_score.toFixed(3)
                  : 'N/A'}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Topics Display */}
      <div className="space-y-4">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Topics and Keywords
        </h3>
        {result.topics.map((topic) => (
          <div key={topic.topic_id} className="space-y-2">
            <div className="flex items-center gap-3">
              <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                <span className="text-sm font-bold text-blue-900 dark:text-blue-100">
                  {topic.topic_id}
                </span>
              </div>
              <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                Topic {topic.topic_id}
              </h4>
            </div>

            {/* Keywords */}
            <p className="text-xs text-gray-600 dark:text-gray-400 ml-11">
              <strong>Keywords:</strong> {topic.keywords}
            </p>

            {/* Top Words Bar Chart */}
            <div className="ml-11 h-48 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={topic.top_words}
                  margin={{ top: 10, right: 20, left: 0, bottom: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                  <XAxis
                    dataKey="word"
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    className="text-gray-700 dark:text-gray-300"
                  />
                  <YAxis className="text-gray-700 dark:text-gray-300" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(31, 41, 55, 0.95)',
                      border: '1px solid rgb(75, 85, 99)',
                      borderRadius: '8px',
                      color: 'white',
                    }}
                    formatter={(value) => (typeof value === 'number' ? value.toFixed(4) : value)}
                  />
                  <Bar dataKey="weight" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </div>

      {/* Summary Stats */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
        <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">
          Model Summary
        </h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-gray-600 dark:text-gray-400">Number of Topics</p>
            <p className="text-lg font-bold text-gray-900 dark:text-gray-100">
              {result.topics.length}
            </p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">Execution Time</p>
            <p className="text-lg font-bold text-gray-900 dark:text-gray-100">
              {result.execution_time_ms.toFixed(0)}ms
            </p>
          </div>
          {result.coherence_score !== undefined && (
            <div>
              <p className="text-gray-600 dark:text-gray-400">Coherence</p>
              <p className="text-lg font-bold text-gray-900 dark:text-gray-100">
                {result.coherence_score.toFixed(4)}
              </p>
            </div>
          )}
          {result.perplexity !== undefined && (
            <div>
              <p className="text-gray-600 dark:text-gray-400">Perplexity</p>
              <p className="text-lg font-bold text-gray-900 dark:text-gray-100">
                {result.perplexity.toFixed(4)}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Info */}
      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <p className="text-xs text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> Each topic is represented by its most frequent
          words. Higher coherence scores indicate more interpretable topics. Words in each
          topic represent the latent themes discovered in the document collection.
        </p>
      </div>
    </div>
  );
}
