import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

interface ClassInfo {
  class_id: number;
  class_name: string;
  color: number[];
  pixel_count: number;
  percentage: number;
  iou_score?: number;
}

interface SegmentationResult {
  success: boolean;
  statistics: {
    total_classes: number;
    total_pixels: number;
    mean_confidence: number;
    class_info: ClassInfo[];
  };
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
}

interface VisualizationProps {
  result: SegmentationResult;
}

export function Visualization({ result }: VisualizationProps) {
  const chartData = (result.statistics.class_info || []).map((cls) => ({
    name: cls.class_name,
    percentage: cls.percentage,
  }));

  return (
    <div className="space-y-4">
      {/* Class Distribution Chart */}
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 20, bottom: 60, left: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
            <XAxis
              dataKey="name"
              angle={-45}
              textAnchor="end"
              height={100}
              interval={0}
              className="text-gray-700 dark:text-gray-300 text-xs"
            />
            <YAxis label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(value) => `${(value as number).toFixed(1)}%`} />
            <Legend />
            <Bar dataKey="percentage" fill="#8b5cf6" name="Class Coverage (%)" fillOpacity={0.8} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Statistics Grid */}
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Segmentation Summary
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <strong>Classes Found:</strong> {result.statistics.total_classes}
            </p>
            <p>
              <strong>Total Pixels:</strong> {(result.statistics.total_pixels / 1e6).toFixed(1)}M
            </p>
            <p>
              <strong>Mean Confidence:</strong>{' '}
              {(result.statistics.mean_confidence * 100).toFixed(1)}%
            </p>
            <p>
              <strong>Inference Time:</strong> {result.execution_time_ms.toFixed(2)} ms
            </p>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Model Configuration
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <strong>Backbone:</strong> {result.parameters_used.model_backbone}
            </p>
            <p>
              <strong>Image Size:</strong> {result.parameters_used.image_size}x
              {result.parameters_used.image_size}
            </p>
            <p>
              <strong>Classes:</strong> {result.parameters_used.num_classes}
            </p>
          </div>
        </div>
      </div>

      {/* Class Details */}
      <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Per-Class Coverage
        </h4>
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {(result.statistics.class_info || []).map((cls, idx) => (
            <div key={idx} className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <div
                  className="w-4 h-4 rounded"
                  style={{
                    backgroundColor: `rgb(${cls.color.join(',')})`,
                  }}
                />
                <span className="text-gray-700 dark:text-gray-300">{cls.class_name}</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-purple-500 h-2 rounded-full"
                    style={{ width: `${cls.percentage}%` }}
                  />
                </div>
                <span className="text-xs text-gray-600 dark:text-gray-400 w-12 text-right">
                  {cls.percentage.toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> Each pixel in the image is classified into one of the
          semantic classes. The chart shows the percentage of image covered by each class. Darker
          colors represent background or less frequent classes.
        </p>
      </div>
    </div>
  );
}
