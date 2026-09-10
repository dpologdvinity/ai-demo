import React from 'react';

interface FaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  center: Record<string, number>;
  area: number;
}

interface FaceDetectionResult {
  success: boolean;
  faces: FaceBox[];
  statistics: {
    face_count: number;
    average_face_size: number;
    largest_face_size: number;
    smallest_face_size: number;
    total_face_area: number;
    face_density: number;
    image_dimensions: Record<string, number>;
    detection_quality: string;
  };
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  algorithm_info: Record<string, any>;
  parameters_used: Record<string, any>;
  image_info: Record<string, any>;
}

interface VisualizationProps {
  result: FaceDetectionResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { visualization_data, statistics, faces } = result;

  return (
    <div className="space-y-4">
      {/* Face Detection Image */}
      {visualization_data.annotated_image && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Detected Faces ({statistics.face_count})
          </h4>
          <img
            src={`data:image/png;base64,${visualization_data.annotated_image}`}
            alt="Faces Detected"
            className="w-full rounded-lg border border-gray-300 dark:border-gray-600"
          />
        </div>
      )}

      {/* Statistics Grid */}
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Face Statistics
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>
              <span className="font-medium">Total Faces:</span> {statistics.face_count}
            </p>
            <p>
              <span className="font-medium">Average Size:</span> {statistics.average_face_size.toFixed(0)} px
            </p>
            <p>
              <span className="font-medium">Largest Face:</span> {statistics.largest_face_size} px
            </p>
            <p>
              <span className="font-medium">Smallest Face:</span> {statistics.smallest_face_size} px
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
              <span className="font-medium">Face Density:</span> {statistics.face_density.toFixed(2)}%
            </p>
            <p>
              <span className="font-medium">Total Face Area:</span> {statistics.total_face_area} px²
            </p>
            <p>
              <span className="font-medium">Quality:</span> {statistics.detection_quality}
            </p>
          </div>
        </div>
      </div>

      {/* Face Details Table */}
      {faces && faces.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Detected Face Coordinates
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="border border-gray-300 dark:border-gray-600 p-2 text-left">
                    Face #
                  </th>
                  <th className="border border-gray-300 dark:border-gray-600 p-2 text-left">X, Y</th>
                  <th className="border border-gray-300 dark:border-gray-600 p-2 text-left">
                    Size
                  </th>
                  <th className="border border-gray-300 dark:border-gray-600 p-2 text-left">
                    Confidence
                  </th>
                </tr>
              </thead>
              <tbody>
                {faces.slice(0, 10).map((face, idx) => (
                  <tr
                    key={idx}
                    className="hover:bg-gray-50 dark:hover:bg-gray-700 border-b border-gray-200 dark:border-gray-700"
                  >
                    <td className="border border-gray-300 dark:border-gray-600 p-2">{idx + 1}</td>
                    <td className="border border-gray-300 dark:border-gray-600 p-2">
                      ({face.x}, {face.y})
                    </td>
                    <td className="border border-gray-300 dark:border-gray-600 p-2">
                      {face.width}x{face.height}
                    </td>
                    <td className="border border-gray-300 dark:border-gray-600 p-2">
                      {(face.confidence * 100).toFixed(0)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {faces.length > 10 && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Showing first 10 of {faces.length} detected faces
            </p>
          )}
        </div>
      )}

      {/* Interpretation */}
      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Interpretation:</strong> Each red rectangle represents a detected face. The detection
          quality depends on lighting, face angle, and occlusions. Adjust parameters to improve detection.
        </p>
      </div>
    </div>
  );
}
