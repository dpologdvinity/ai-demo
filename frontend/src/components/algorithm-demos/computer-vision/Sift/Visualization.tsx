import React from 'react';

interface KeypointData {
  x: number;
  y: number;
  size: number;
  angle: number;
  response: number;
  octave: number;
}

interface SiftResult {
  success: boolean;
  statistics: {
    keypoint_count: number;
    image_dimensions: Record<string, number>;
    average_scale: number;
    average_response: number;
    scale_distribution: Record<string, number>;
    orientation_distribution: Record<string, number>;
    octave_distribution: Record<string, number>;
    descriptor_dimensions: number;
  };
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  algorithm_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
  keypoints: KeypointData[];
  match_statistics?: {
    match_count: number;
    total_matches: number;
    keypoints_image1: number;
    keypoints_image2: number;
    match_ratio: number;
    average_match_distance: number;
  };
}

interface VisualizationProps {
  result: SiftResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { visualization_data, statistics, keypoints, match_statistics } = result;

  return (
    <div className="space-y-4">
      {/* Keypoints Visualization */}
      {visualization_data.annotated_image && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Detected Keypoints ({statistics.keypoint_count})
          </h4>
          <img
            src={`data:image/png;base64,${visualization_data.annotated_image}`}
            alt="SIFT Keypoints"
            className="w-full rounded-lg border border-gray-300 dark:border-gray-600"
          />
        </div>
      )}

      {/* Matches Visualization (if in matching mode) */}
      {match_statistics && visualization_data.match_image && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Feature Matches ({match_statistics.match_count})
          </h4>
          <img
            src={`data:image/png;base64,${visualization_data.match_image}`}
            alt="Feature Matches"
            className="w-full rounded-lg border border-gray-300 dark:border-gray-600"
          />
        </div>
      )}

      {/* Statistics Grid */}
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Feature Statistics
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <span className="font-medium">Keypoints:</span> {statistics.keypoint_count}
            </p>
            <p>
              <span className="font-medium">Avg Scale:</span> {statistics.average_scale.toFixed(2)}
            </p>
            <p>
              <span className="font-medium">Avg Response:</span> {statistics.average_response.toFixed(3)}
            </p>
            <p>
              <span className="font-medium">Descriptor Dims:</span> {statistics.descriptor_dimensions}
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
              <span className="font-medium">Octaves:</span> {Object.keys(statistics.octave_distribution).length}
            </p>
            <p>
              <span className="font-medium">Contrast Thresh:</span>{' '}
              {result.parameters_used.contrastThreshold}
            </p>
            <p>
              <span className="font-medium">Edge Thresh:</span> {result.parameters_used.edgeThreshold}
            </p>
          </div>
        </div>
      </div>

      {/* Match Statistics (if available) */}
      {match_statistics && (
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">
            Feature Matching Results
          </h4>
          <div className="space-y-1 text-xs text-purple-800 dark:text-purple-200">
            <p>
              <span className="font-medium">Good Matches:</span> {match_statistics.match_count}
            </p>
            <p>
              <span className="font-medium">Total Matches:</span> {match_statistics.total_matches}
            </p>
            <p>
              <span className="font-medium">Image 1 Keypoints:</span> {match_statistics.keypoints_image1}
            </p>
            <p>
              <span className="font-medium">Image 2 Keypoints:</span> {match_statistics.keypoints_image2}
            </p>
            <p>
              <span className="font-medium">Match Ratio:</span> {(match_statistics.match_ratio * 100).toFixed(1)}
              %
            </p>
            <p>
              <span className="font-medium">Avg Match Distance:</span>{' '}
              {match_statistics.average_match_distance.toFixed(2)}
            </p>
          </div>
        </div>
      )}

      {/* Keypoints Table */}
      {keypoints && keypoints.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Top Keypoints (Strongest Responses)
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="border border-gray-300 dark:border-gray-600 p-2 text-left">ID</th>
                  <th className="border border-gray-300 dark:border-gray-600 p-2 text-left">X, Y</th>
                  <th className="border border-gray-300 dark:border-gray-600 p-2 text-left">
                    Size
                  </th>
                  <th className="border border-gray-300 dark:border-gray-600 p-2 text-left">
                    Angle
                  </th>
                  <th className="border border-gray-300 dark:border-gray-600 p-2 text-left">
                    Response
                  </th>
                </tr>
              </thead>
              <tbody>
                {keypoints.slice(0, 10).map((kp, idx) => (
                  <tr
                    key={idx}
                    className="hover:bg-gray-50 dark:hover:bg-gray-700 border-b border-gray-200 dark:border-gray-700"
                  >
                    <td className="border border-gray-300 dark:border-gray-600 p-2">{idx + 1}</td>
                    <td className="border border-gray-300 dark:border-gray-600 p-2">
                      ({kp.x.toFixed(1)}, {kp.y.toFixed(1)})
                    </td>
                    <td className="border border-gray-300 dark:border-gray-600 p-2">
                      {kp.size.toFixed(1)}
                    </td>
                    <td className="border border-gray-300 dark:border-gray-600 p-2">
                      {kp.angle.toFixed(1)}°
                    </td>
                    <td className="border border-gray-300 dark:border-gray-600 p-2">
                      {kp.response.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {keypoints.length > 10 && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Showing first 10 of {keypoints.length} keypoints
            </p>
          )}
        </div>
      )}

      {/* Interpretation */}
      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> Circles show detected keypoints; circle size represents the
          scale at which the feature was detected. These features are invariant to rotation and scale,
          making them ideal for image matching and object recognition.
        </p>
      </div>
    </div>
  );
}
