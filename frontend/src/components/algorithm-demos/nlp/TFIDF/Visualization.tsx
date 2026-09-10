import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  feature_names: string[];
  top_terms_global: Array<{ term: string; score: number }>;
  metrics: Record<string, any>;
  tfidf_matrix: number[][];
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const chartData = useMemo(() => {
    return result.top_terms_global.slice(0, 20).map((item) => ({
      term: item.term,
      score: parseFloat(item.score.toFixed(4)),
    }));
  }, [result.top_terms_global]);

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
            <YAxis label={{ value: 'TF-IDF Score', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--background)',
                border: '1px solid var(--border)',
              }}
            />
            <Bar dataKey="score" fill="#f59e0b" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardContent className="pt-6">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Feature Statistics
            </h4>
            <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
              <p>Total Features: {result.feature_names.length}</p>
              <p>Documents: {result.tfidf_matrix.length}</p>
              <p>Sparsity: {result.metrics.sparsity?.toFixed(2)}%</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Top Term
            </h4>
            <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
              <p className="font-mono font-bold text-amber-600 dark:text-amber-400">
                {result.top_terms_global[0]?.term}
              </p>
              <p>Score: {result.top_terms_global[0]?.score?.toFixed(4)}</p>
              <p>Terms Shown: 20</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> TF-IDF scores combine word frequency in documents (TF)
          with rarity across the corpus (IDF). Higher scores indicate terms that are important in specific
          documents but not common across all documents.
        </p>
      </div>
    </div>
  );
}
