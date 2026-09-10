import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface TaggedWord {
  text: string;
  tag: string;
  pos_fine: string;
  pos_coarse: string;
  description: string;
}

interface TrainingResult {
  success: boolean;
  tagged_words: TaggedWord[];
  pos_distribution: Record<string, number>;
  execution_time_ms: number;
  text_analyzed: string;
}

interface VisualizationProps {
  result: TrainingResult;
}

const TAG_COLORS: Record<string, string> = {
  'NOUN': '#3b82f6',
  'VERB': '#10b981',
  'ADJ': '#f59e0b',
  'ADV': '#8b5cf6',
  'PRON': '#ec4899',
  'DET': '#6366f1',
  'PREP': '#14b8a6',
  'CONJ': '#f97316',
  'PUNCT': '#9ca3af',
  'NUM': '#06b6d4',
  'PRT': '#84cc16',
  'X': '#64748b',
};

const TAG_DESCRIPTIONS: Record<string, string> = {
  'NOUN': 'Noun',
  'VERB': 'Verb',
  'ADJ': 'Adjective',
  'ADV': 'Adverb',
  'PRON': 'Pronoun',
  'DET': 'Determiner',
  'PREP': 'Preposition',
  'CONJ': 'Conjunction',
  'PUNCT': 'Punctuation',
  'NUM': 'Number',
  'PRT': 'Particle',
  'X': 'Other',
};

export function Visualization({ result }: VisualizationProps) {
  // Prepare data for bar chart
  const chartData = Object.entries(result.pos_distribution)
    .map(([tag, count]) => ({
      tag,
      count,
    }))
    .sort((a, b) => b.count - a.count);

  // Get color for a tag
  const getTagColor = (tag: string) => {
    const baseTag = tag.split('_')[0];
    return TAG_COLORS[baseTag] || TAG_COLORS['X'];
  };

  return (
    <div className="space-y-6">
      {/* Tagged Text Display */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Analyzed Text
        </h3>
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="flex flex-wrap gap-2">
            {result.tagged_words.map((word, idx) => (
              <div
                key={idx}
                className="flex flex-col items-center"
                title={word.description}
              >
                <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  {word.text}
                </span>
                <span
                  className="text-xs font-semibold px-2 py-1 rounded text-white mt-1"
                  style={{ backgroundColor: getTagColor(word.pos_coarse) }}
                >
                  {word.tag}
                </span>
              </div>
            ))}
          </div>
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400">
          {result.text_analyzed}
        </p>
      </div>

      {/* POS Distribution Chart */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          POS Distribution
        </h3>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
              <XAxis
                dataKey="tag"
                angle={-45}
                textAnchor="end"
                height={100}
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
              />
              <Bar
                dataKey="count"
                fill="#3b82f6"
                radius={[8, 8, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Tag Details Table */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Tag Reference
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left px-3 py-2 text-gray-700 dark:text-gray-300">Tag</th>
                <th className="text-left px-3 py-2 text-gray-700 dark:text-gray-300">Count</th>
                <th className="text-left px-3 py-2 text-gray-700 dark:text-gray-300">Description</th>
              </tr>
            </thead>
            <tbody>
              {chartData.map(({ tag, count }) => (
                <tr key={tag} className="border-b border-gray-100 dark:border-gray-800">
                  <td className="px-3 py-2">
                    <span
                      className="inline-block px-2 py-1 rounded text-white text-xs font-semibold"
                      style={{ backgroundColor: getTagColor(tag) }}
                    >
                      {tag}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-gray-600 dark:text-gray-400">{count}</td>
                  <td className="px-3 py-2 text-gray-600 dark:text-gray-400">
                    {TAG_DESCRIPTIONS[tag.split('_')[0]] || tag}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Metrics Summary */}
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400">Words Tagged</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {result.tagged_words.length}
              </p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400">Execution Time</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {result.execution_time_ms.toFixed(1)}ms
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
