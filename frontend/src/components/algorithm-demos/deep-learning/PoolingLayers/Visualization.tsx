import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface PoolingResult {
  success: boolean;
  input_feature_map: number[][][];
  pooled_output: number[][][];
  dimension_info: Record<string, any>;
  pooling_windows: any[];
  comparison_data: Record<string, any>;
}

interface VisualizationProps {
  result: PoolingResult;
}

export function Visualization({ result }: VisualizationProps) {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Dimensions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-xs">
          <div>
            <p className="font-medium mb-1">Input Shape:</p>
            <p className="text-gray-600 dark:text-gray-400">
              {result.dimension_info?.input_shape?.join(' × ')}
            </p>
          </div>
          <div>
            <p className="font-medium mb-1">Output Shape:</p>
            <p className="text-gray-600 dark:text-gray-400">
              {result.dimension_info?.output_shape?.join(' × ')}
            </p>
          </div>
          <div>
            <p className="font-medium mb-1">Reduction Factor:</p>
            <p className="text-gray-600 dark:text-gray-400">
              {result.dimension_info?.reduction_factor}x
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Pooling Windows</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-gray-600 dark:text-gray-400">
          <p className="mb-2">
            The pooling operation is applied to sliding windows of the input feature map.
            Each window is reduced to a single value based on the pooling type:
          </p>
          <ul className="list-disc list-inside space-y-1">
            <li><strong>Max:</strong> Takes maximum value in window</li>
            <li><strong>Average:</strong> Takes mean of values in window</li>
            <li><strong>Global:</strong> Reduces entire feature map to single value</li>
          </ul>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Comparison</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-gray-600 dark:text-gray-400">
          <p>
            Max pooling preserves peak activations and is commonly used in CNNs for its effectiveness.
            Average pooling provides a smoother representation. Global pooling reduces spatial dimensions
            completely, useful for transitioning to fully connected layers.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">About Pooling</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-gray-600 dark:text-gray-400">
          <p>
            Pooling layers reduce spatial dimensions while preserving important features,
            decreasing computation and memory usage while maintaining feature robustness.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
