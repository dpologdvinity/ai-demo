import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  vocabulary: string[];
  most_frequent_terms: Array<{ term: string; frequency: number }>;
  metrics: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const chartData = useMemo(() => {
    return result.most_frequent_terms.slice(0, 20).map((item) => ({
      term: item.term,
      frequency: item.frequency,
    }));
  }, [result.most_frequent_terms]);

  return (
    <div className="space-y-4">
      <div className="h-[400px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
            <XAxis
              dataKey="term"
              angle={-45}
              textAnchor="end"
              height={100}
              interval={0}
              tick={{ fontSize: 12 }}
            />
            <YAxis label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--background)',
                border: '1px solid var(--border)',
              }}
            />
            <Bar dataKey="frequency" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardContent className="pt-6">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Vocabulary Stats
            </h4>
            <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
              <p>Vocabulary Size: {result.vocabulary.length}</p>
              <p>Sparsity: {result.metrics.sparsity?.toFixed(2)}%</p>
              <p>Avg Terms/Doc: {result.metrics.avg_terms_per_doc?.toFixed(2)}</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Most Frequent Terms
            </h4>
            <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
              <p>Top Term: {result.most_frequent_terms[0]?.term}</p>
              <p>Max Frequency: {result.most_frequent_terms[0]?.frequency}</p>
              <p>Terms Shown: 20</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> The chart shows the top 20 most frequently occurring terms
          in the document collection. Higher bars indicate words that appear more often across documents.
        </p>
      </div>
    </div>
  );
}
