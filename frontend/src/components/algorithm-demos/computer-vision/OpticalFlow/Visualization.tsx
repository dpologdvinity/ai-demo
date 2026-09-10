import React from 'react';

interface FlowVector {
  x: number;
  y: number;
  dx: number;
  dy: number;
  magnitude: number;
  angle: number;
}

interface OpticalFlowResult {
  success: boolean;
  statistics: {
    average_magnitude: number;
    max_magnitude: number;
    min_magnitude: number;
    median_magnitude: number;
    flow_coverage: number;
    primary_direction: number;
    image_dimensions: Record<string, number>;
    total_pixels: number;
  };
  visualization_data: Record<string, any>;
  flow_vectors: FlowVector[];
  execution_time_ms: number;
  algorithm_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
}

interface VisualizationProps {
  result: OpticalFlowResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { visualization_data, statistics, flow_vectors } = result;

  return (
    <div className="space-y-4">
      {/* Flow Visualization */}
      {visualization_data.flow_visualization && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Optical Flow Field
          </h4>
          <img
            src={`data:image/png;base64,${visualization_data.flow_visualization}`}
            alt="Optical Flow"
            className="w-full rounded-lg border border-gray-300 dark:border-gray-600"
          />
        </div>
      )}

      {/* Flow Magnitude Heatmap */}
      {visualization_data.magnitude_map && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Motion Magnitude Heatmap
          </h4>
          <img
            src={`data:image/png;base64,${visualization_data.magnitude_map}`}
            alt="Motion Magnitude"
            className="w-full rounded-lg border border-gray-300 dark:border-gray-600"
          />
        </div>
      )}

      {/* Statistics Grid */}
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Motion Statistics
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <span className="font-medium">Avg Magnitude:</span>{' '}
              {statistics.average_magnitude.toFixed(2)}
            </p>
            <p>
              <span className="font-medium">Max Magnitude:</span> {statistics.max_magnitude.toFixed(2)}
            </p>
            <p>
              <span className="font-medium">Min Magnitude:</span> {statistics.min_magnitude.toFixed(2)}
            </p>
            <p>
              <span className="font-medium">Median Magnitude:</span>{' '}
              {statistics.median_magnitude.toFixed(2)}
            </p>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Flow Information
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <span className="font-medium">Coverage:</span> {statistics.flow_coverage.toFixed(2)}%
            </p>
            <p>
              <span className="font-medium">Primary Direction:</span>{' '}
              {statistics.primary_direction.toFixed(1)}°
            </p>
            <p>
              <span className="font-medium">Method:</span> {result.parameters_used.method}
            </p>
            <p>
              <span className="font-medium">Pyramid Levels:</span> {result.parameters_used.levels}
            </p>
          </div>
        </div>
      </div>

      {/* Image Dimensions */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
          Frame Information
        </h4>
        <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
          <p>
            <span className="font-medium">Dimensions:</span> {statistics.image_dimensions.width}x
            {statistics.image_dimensions.height}
          </p>
          <p>
            <span className="font-medium">Total Pixels:</span> {statistics.total_pixels}
          </p>
          <p>
            <span className="font-medium">Window Size:</span> {result.parameters_used.winsize}
          </p>
          <p>
            <span className="font-medium">Iterations:</span> {result.parameters_used.iterations}
          </p>
        </div>
      </div>

      {/* Flow Vectors Table */}
      {flow_vectors && flow_vectors.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Sample Flow Vectors
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="border border-gray-300 dark:border-gray-600 p-2 text-left">
                    Location
                  </th>
                  <th className="border border-gray-300 dark:border-gray-600 p-2 text-left">
                    Motion (dx, dy)
                  </th>
                  <th className="border border-gray-300 dark:border-gray-600 p-2 text-left">
                    Magnitude
                  </th>
                  <th className="border border-gray-300 dark:border-gray-600 p-2 text-left">
                    Direction
                  </th>
                </tr>
              </thead>
              <tbody>
                {flow_vectors.slice(0, 10).map((vec, idx) => (
                  <tr
                    key={idx}
                    className="hover:bg-gray-50 dark:hover:bg-gray-700 border-b border-gray-200 dark:border-gray-700"
                  >
                    <td className="border border-gray-300 dark:border-gray-600 p-2">
                      ({vec.x}, {vec.y})
                    </td>
                    <td className="border border-gray-300 dark:border-gray-600 p-2">
                      ({vec.dx.toFixed(2)}, {vec.dy.toFixed(2)})
                    </td>
                    <td className="border border-gray-300 dark:border-gray-600 p-2">
                      {vec.magnitude.toFixed(2)}
                    </td>
                    <td className="border border-gray-300 dark:border-gray-600 p-2">
                      {vec.angle.toFixed(1)}°
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {flow_vectors.length > 10 && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Showing first 10 of {flow_vectors.length} flow vectors
            </p>
          )}
        </div>
      )}

      {/* Interpretation */}
      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> Colors represent flow magnitude (darker = more motion). Arrows
          or dots show motion direction. Higher flow coverage indicates more pixels with significant
          motion between frames. Primary direction shows dominant motion in the scene.
        </p>
      </div>
    </div>
  );
}
