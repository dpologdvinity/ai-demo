import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent } from '@/components/ui/card';

interface TrainingResult {
  success: boolean;
  original_text: string;
  text_preview: string;
  main_result?: {
    tokens: any[];
    token_count: number;
    unique_token_count: number;
    token_frequency: Array<{ token: string; count: number }>;
  };
  comparison_results: any[];
  statistics: Record<string, any>;
  frequency_distribution: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const mainResult = result.main_result;

  const frequencyChartData = useMemo(() => {
    if (!mainResult?.token_frequency) return [];
    return mainResult.token_frequency.slice(0, 15).map((item: any) => ({
      token: item.token || item.word,
      count: item.count || item.frequency,
    }));
  }, [mainResult]);

  const tokenList = useMemo(() => {
    if (!mainResult?.tokens) return [];
    return mainResult.tokens.slice(0, 30);
  }, [mainResult]);

  return (
    <div className="space-y-4">
      {/* Frequency Chart */}
      {frequencyChartData.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
            Token Frequency Distribution
          </h4>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={frequencyChartData} margin={{ top: 20, right: 30, left: 0, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
                <XAxis
                  dataKey="token"
                  angle={-45}
                  textAnchor="end"
                  height={100}
                  tick={{ fontSize: 12 }}
                />
                <YAxis label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--background)',
                    border: '1px solid var(--border)',
                  }}
                />
                <Bar dataKey="count" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Token List */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Tokens (First 30)
        </h4>
        <div className="grid grid-cols-2 gap-2">
          {tokenList.map((token: any, idx: number) => (
            <div key={idx} className="p-2 bg-gray-50 dark:bg-gray-800 rounded text-xs">
              <span className="font-mono text-gray-700 dark:text-gray-300">{token.text || token}</span>
              {token.is_stopword && (
                <span className="ml-1 text-xs text-red-600 dark:text-red-400">(stopword)</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-3 gap-3">
        <Card>
          <CardContent className="pt-4">
            <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {mainResult?.token_count || 0}
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Total Tokens</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
              {mainResult?.unique_token_count || 0}
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Unique Tokens</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <p className="text-2xl font-bold text-green-600 dark:text-green-400">
              {(mainResult as any)?.avg_token_length ? ((mainResult as any).avg_token_length as number).toFixed(1) : '0'}
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Avg Length</p>
          </CardContent>
        </Card>
      </div>

      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Original Text:</strong> {result.text_preview}
        </p>
      </div>
    </div>
  );
}
