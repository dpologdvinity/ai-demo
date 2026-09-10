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

interface Entity {
  text: string;
  label: string;
  start: number;
  end: number;
  confidence: number;
}

interface TrainingResult {
  success: boolean;
  entities: Entity[];
  entity_distribution: Record<string, number>;
  execution_time_ms: number;
}

interface VisualizationProps {
  result: TrainingResult;
}

const ENTITY_COLORS: Record<string, string> = {
  'PERSON': '#ef4444',
  'ORG': '#f59e0b',
  'GPE': '#3b82f6',
  'DATE': '#8b5cf6',
  'TIME': '#ec4899',
  'MONEY': '#10b981',
  'PERCENT': '#06b6d4',
  'FACILITY': '#f97316',
  'PRODUCT': '#6366f1',
  'EVENT': '#14b8a6',
  'LAW': '#84cc16',
  'LANGUAGE': '#64748b',
  'NORP': '#eab308',
  'CARDINAL': '#a855f7',
  'ORDINAL': '#d946ef',
  'QUANTITY': '#22d3ee',
};

export function Visualization({ result }: VisualizationProps) {
  // Sort entities by label for chart
  const chartData = Object.entries(result.entity_distribution)
    .map(([label, count]) => ({
      label,
      count,
    }))
    .sort((a, b) => b.count - a.count);

  const getEntityColor = (label: string) => {
    return ENTITY_COLORS[label] || '#9ca3af';
  };

  return (
    <div className="space-y-6">
      {/* Entities List */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Extracted Entities
        </h3>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {result.entities.length === 0 ? (
            <p className="text-sm text-gray-500 dark:text-gray-400">No entities found.</p>
          ) : (
            result.entities.map((entity, idx) => (
              <div
                key={idx}
                className="p-3 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1">
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      {entity.text}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Position: {entity.start}-{entity.end}
                    </p>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <span
                      className="text-xs font-semibold px-2 py-1 rounded text-white"
                      style={{ backgroundColor: getEntityColor(entity.label) }}
                    >
                      {entity.label}
                    </span>
                    {entity.confidence < 1.0 && (
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {(entity.confidence * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Distribution Chart */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Entity Type Distribution
        </h3>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
              <XAxis
                dataKey="label"
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
              <Bar dataKey="count" fill="#3b82f6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Entities</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {result.entities.length}
              </p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400">Entity Types</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {Object.keys(result.entity_distribution).length}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
