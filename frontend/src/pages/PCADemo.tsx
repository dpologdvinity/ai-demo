import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiService } from '@/services/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';
import { ParameterControl } from '@/components/common/ParameterControl';
import Button from '@/components/common/Button';
import { ScatterPlot } from '@/components/visualizations/ScatterPlot';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { ErrorDisplay } from '@/components/common/ErrorDisplay';

interface PCAParameters {
  n_components: number;
  whiten: boolean;
  random_state: number;
}

interface PCAResponse {
  success: boolean;
  transformed_data: number[][];
  labels: number[];
  explained_variance: number[];
  explained_variance_ratio: number[];
  cumulative_variance_ratio: number[];
  visualization_data: {
    scatter_data: Array<{
      pc1: number;
      pc2: number;
      pc3?: number;
      label: number;
    }>;
    variance_data: Array<{
      component: string;
      variance: number;
      cumulative: number;
    }>;
    n_samples: number;
    n_features_original: number;
  };
  execution_time_ms: number;
  model_info: {
    n_components: number;
    whiten: boolean;
    n_features: number;
    n_samples: number;
  };
  parameters_used: PCAParameters;
  error?: string;
}

function PCADemo() {
  const [params, setParams] = useState<PCAParameters>({
    n_components: 2,
    whiten: false,
    random_state: 42,
  });

  const mutation = useMutation({
    mutationFn: (parameters: PCAParameters) =>
      apiService.trainAlgorithm('ml', 'pca', parameters),
  });

  const handleTrain = () => {
    mutation.mutate(params);
  };

  const handleParamChange = (key: keyof PCAParameters, value: number | boolean) => {
    setParams((prev) => ({ ...prev, [key]: value }));
  };

  const result = mutation.data as PCAResponse | undefined;

  return (
    <div className="container mx-auto py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold mb-2">Principal Component Analysis (PCA)</h1>
        <p className="text-lg text-muted-foreground">
          Dimensionality reduction technique that finds principal components to visualize high-dimensional data
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Parameters Panel */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Parameters</CardTitle>
              <CardDescription>Configure PCA parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <ParameterControl
                label="Number of Components"
                value={params.n_components}
                onChange={(v) => handleParamChange('n_components', Number(v))}
                type="slider"
                min={2}
                max={10}
                step={1}
                description="Number of principal components to compute"
              />

              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Whiten Components</label>
                <input
                  type="checkbox"
                  checked={params.whiten}
                  onChange={(e) => handleParamChange('whiten', e.target.checked)}
                  className="h-4 w-4"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Divide components by singular values for unit variance
              </p>

              <Button
                onClick={handleTrain}
                disabled={mutation.isPending}
                className="w-full"
              >
                {mutation.isPending ? 'Training...' : 'Train PCA'}
              </Button>

              {mutation.isPending && (
                <div className="flex justify-center">
                  <LoadingSpinner size="sm" />
                </div>
              )}
            </CardContent>
          </Card>

          {/* Dataset Info */}
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Dataset</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-sm space-y-2">
                <p><strong>Name:</strong> Digits Dataset</p>
                <p><strong>Samples:</strong> 1,797 handwritten digits</p>
                <p><strong>Original Features:</strong> 64 (8x8 pixel images)</p>
                <p><strong>Classes:</strong> 10 (digits 0-9)</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {mutation.isError && (
            <ErrorDisplay message="Failed to train PCA. Please try again." />
          )}

          {result && result.success && (
            <>
              {/* Metrics */}
              <Card>
                <CardHeader>
                  <CardTitle>Results</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Components</p>
                      <p className="text-2xl font-bold">{result.model_info.n_components}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Samples</p>
                      <p className="text-2xl font-bold">{result.visualization_data.n_samples}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Variance Explained</p>
                      <p className="text-2xl font-bold">
                        {(result.cumulative_variance_ratio[result.cumulative_variance_ratio.length - 1] * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Time</p>
                      <p className="text-2xl font-bold">{result.execution_time_ms.toFixed(0)}ms</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Scatter Plot */}
              <Card>
                <CardHeader>
                  <CardTitle>Principal Components Visualization</CardTitle>
                  <CardDescription>
                    Data projected onto the first {result.model_info.n_components} principal components
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ScatterPlot
                    data={result.visualization_data.scatter_data}
                    xKey="pc1"
                    yKey="pc2"
                    colorKey="label"
                    xLabel="First Principal Component"
                    yLabel="Second Principal Component"
                    height={400}
                    showGrid={true}
                    showLegend={true}
                    clusterNames={['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']}
                  />
                </CardContent>
              </Card>

              {/* Explained Variance */}
              <Card>
                <CardHeader>
                  <CardTitle>Explained Variance Ratio</CardTitle>
                  <CardDescription>
                    Proportion of variance explained by each principal component
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={result.visualization_data.variance_data}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis dataKey="component" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="variance" fill="#8884d8" name="Variance Ratio" />
                      <Bar dataKey="cumulative" fill="#82ca9d" name="Cumulative Variance" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Theory */}
              <Card>
                <CardHeader>
                  <CardTitle>About PCA</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h3 className="font-semibold mb-2">How it works</h3>
                    <p className="text-sm text-muted-foreground">
                      PCA transforms high-dimensional data into a new coordinate system where axes (principal components)
                      are ordered by variance. The first component captures the most variance, the second captures the
                      second most (orthogonal to the first), and so on.
                    </p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Use Cases</h3>
                    <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                      <li>Data visualization (reducing to 2D or 3D)</li>
                      <li>Noise reduction and feature extraction</li>
                      <li>Speeding up machine learning algorithms</li>
                      <li>Exploratory data analysis</li>
                    </ul>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Complexity</h3>
                    <p className="text-sm text-muted-foreground">
                      <strong>Time:</strong> O(min(n²×d, d²×n)) where n is samples and d is features<br />
                      <strong>Space:</strong> O(n×d)
                    </p>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default PCADemo;
