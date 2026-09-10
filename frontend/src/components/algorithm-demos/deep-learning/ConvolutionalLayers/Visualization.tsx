import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface ConvResult {
  success: boolean;
  input_image: number[][];
  filter_kernels: any[];
  feature_maps: any[];
  common_filters: Record<string, any>;
  output_dimensions: Record<string, any>;
  activation_stats: Record<string, any>;
  visualization_data: Record<string, any>;
}

interface VisualizationProps {
  result: ConvResult;
}

export function Visualization({ result }: VisualizationProps) {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Convolution Operation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-xs">
          <div>
            <p className="font-medium mb-2">Input Image Shape:</p>
            <p className="text-gray-600 dark:text-gray-400">8 x 8 x 1</p>
          </div>
          <div>
            <p className="font-medium mb-2">Output Feature Maps Shape:</p>
            <p className="text-gray-600 dark:text-gray-400">
              {result.output_dimensions?.height} x {result.output_dimensions?.width} x{' '}
              {result.output_dimensions?.channels}
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Activation Statistics</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          <div>
            <span className="font-medium">Mean Activation:</span>{' '}
            {result.activation_stats?.mean_activation?.toFixed(4)}
          </div>
          <div>
            <span className="font-medium">Max Activation:</span>{' '}
            {result.activation_stats?.max_activation?.toFixed(4)}
          </div>
          <div>
            <span className="font-medium">Min Activation:</span>{' '}
            {result.activation_stats?.min_activation?.toFixed(4)}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Common Filters Applied</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-gray-600 dark:text-gray-400">
          <p className="mb-2">
            Standard edge detection filters (Sobel, etc.) have been applied to demonstrate how different
            filter weights extract different features from the same input.
          </p>
          <ul className="list-disc list-inside space-y-1">
            <li>Sobel-X: Detects vertical edges</li>
            <li>Sobel-Y: Detects horizontal edges</li>
            <li>Gaussian: Blurs and smooths features</li>
          </ul>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">About Convolution</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-gray-600 dark:text-gray-400">
          <p>
            Convolution applies learned filters across the input image, with each filter detecting specific
            patterns or features. The output for each filter is a feature map. By learning many filters, the
            network can extract diverse and hierarchical features.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
