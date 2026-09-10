import React from 'react';

interface EdgeDetectionResult {
  success: boolean;
  statistics: {
    edge_pixel_count: number;
    total_pixels: number;
    edge_density: number;
    image_dimensions: Record<string, number>;
    threshold_ratio: number;
  };
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  algorithm_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
}

interface VisualizationProps {
  result: EdgeDetectionResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { visualization_data, statistics, image_info, parameters_used } = result;

  return (
    <div className="space-y-4">
      {/* Edge Detection Visualization */}
      <div className="space-y-2">
        {visualization_data.original_image && (
          <div>
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Original Image
            </h4>
            <img
              src={`data:image/png;base64,${visualization_data.original_image}`}
              alt="Original"
              className="w-full rounded-lg border border-gray-300 dark:border-gray-600"
            />
          </div>
        )}

        {visualization_data.detected_edges && (
          <div>
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Detected Edges
            </h4>
            <img
              src={`data:image/png;base64,${visualization_data.detected_edges}`}
              alt="Detected Edges"
              className="w-full rounded-lg border border-gray-300 dark:border-gray-600"
            />
          </div>
        )}
      </div>

      {/* Statistics Grid */}
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Detection Statistics
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <span className="font-medium">Edge Pixels:</span> {statistics.edge_pixel_count}
            </p>
            <p>
              <span className="font-medium">Total Pixels:</span> {statistics.total_pixels}
            </p>
            <p>
              <span className="font-medium">Edge Density:</span> {statistics.edge_density.toFixed(2)}%
            </p>
            <p>
              <span className="font-medium">Threshold Ratio:</span> {statistics.threshold_ratio.toFixed(2)}
            </p>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Image Information
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <span className="font-medium">Dimensions:</span> {statistics.image_dimensions.width}x
              {statistics.image_dimensions.height}
            </p>
            <p>
              <span className="font-medium">Threshold1:</span> {parameters_used.threshold1}
            </p>
            <p>
              <span className="font-medium">Threshold2:</span> {parameters_used.threshold2}
            </p>
            <p>
              <span className="font-medium">Aperture:</span> {parameters_used.aperture_size}x
              {parameters_used.aperture_size}
            </p>
          </div>
        </div>
      </div>

      {/* Interpretation */}
      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> The white pixels in the edge detection image represent detected
          edges. Edge density indicates what percentage of the image consists of edges. Higher edge density
          suggests a complex or textured image with many features.
        </p>
      </div>
    </div>
  );
}
