import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface TrainingResult {
  success: boolean;
  instances: any[];
  statistics: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const classChartData = Object.entries(result.statistics.class_counts || {}).map(
    ([className, count]) => ({
      class: className,
      count: count,
    })
  );

  return (
    <div className="space-y-4">
      {/* Class Distribution Chart */}
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={classChartData}
            margin={{ top: 20, right: 20, bottom: 60, left: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
            <XAxis
              dataKey="class"
              angle={-45}
              textAnchor="end"
              height={100}
              interval={0}
              className="text-gray-700 dark:text-gray-300 text-xs"
            />
            <YAxis className="text-gray-700 dark:text-gray-300" />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" fill="#10b981" name="Instance Count" fillOpacity={0.8} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Statistics Grid */}
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Detection Summary
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <strong>Total Instances:</strong> {result.statistics.total_instances}
            </p>
            <p>
              <strong>Unique Classes:</strong> {result.statistics.unique_classes}
            </p>
            <p>
              <strong>Coverage:</strong> {result.statistics.coverage_percentage.toFixed(1)}%
            </p>
            <p>
              <strong>Avg Confidence:</strong>{' '}
              {(result.statistics.avg_confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Mask Statistics
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <strong>Total Mask Area:</strong>{' '}
              {(result.statistics.total_mask_area / 1e4).toFixed(1)}K pixels
            </p>
            <p>
              <strong>Avg Instance Size:</strong>{' '}
              {(result.statistics.avg_instance_size / 1e3).toFixed(1)}K pixels
            </p>
            <p>
              <strong>Inference Time:</strong> {result.execution_time_ms.toFixed(2)} ms
            </p>
          </div>
        </div>
      </div>

      {/* Confidence Distribution */}
      {result.statistics.confidence_distribution && (
        <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
            Confidence Distribution
          </h4>
          <div className="space-y-2">
            {Object.entries(result.statistics.confidence_distribution as Record<string, number>).map(([range, count]) => {
              const maxCount = Math.max(...Object.values(result.statistics.confidence_distribution as Record<string, number>));
              return (
                <div key={range} className="flex items-center justify-between text-sm">
                  <span className="text-gray-700 dark:text-gray-300">{range}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-green-500 h-2 rounded-full"
                        style={{
                          width: `${(count / maxCount) * 100}%`,
                        }}
                      />
                    </div>
                    <span className="text-xs text-gray-600 dark:text-gray-400 w-8 text-right">
                      {count}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> Instance segmentation detects individual objects and
          creates pixel-level masks for each. The chart shows the distribution of detected object
          classes. Masks indicate the exact pixels belonging to each instance.
        </p>
      </div>
    </div>
  );
}
