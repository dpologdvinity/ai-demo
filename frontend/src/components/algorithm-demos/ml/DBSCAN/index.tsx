import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { AlgorithmLayout } from '@/components/common/AlgorithmLayout';
import { apiService } from '@/services/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';
import Controls from './Controls';
import Visualization from './Visualization';
import Documentation from './Documentation';
import { Loader2 } from 'lucide-react';

interface DBSCANParameters {
  eps: number;
  min_samples: number;
  metric: string;
  random_state: number;
  n_samples: number;
  noise: number;
}

interface ClusterInfo {
  cluster_id: number;
  size: number;
  is_noise: boolean;
}

interface DBSCANResult {
  success: boolean;
  metrics: {
    n_clusters: number;
    n_noise: number;
    noise_ratio: number;
    silhouette_score?: number;
  };
  clusters: ClusterInfo[];
  visualization_data: {
    clusters: Array<Array<{ x: number; y: number; cluster: number }>>;
    noise_points: Array<{ x: number; y: number; cluster: number }>;
    n_clusters: number;
  };
  execution_time_ms: number;
  parameters_used: DBSCANParameters;
  error?: string;
}

function DBSCANClustering() {
  const [parameters, setParameters] = useState<DBSCANParameters>({
    eps: 0.5,
    min_samples: 5,
    metric: 'euclidean',
    random_state: 42,
    n_samples: 300,
    noise: 0.1,
  });

  const [result, setResult] = useState<DBSCANResult | null>(null);

  const trainMutation = useMutation({
    mutationFn: (params: DBSCANParameters) =>
      apiService.post<DBSCANResult>('/api/ml/dbscan/train', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof DBSCANParameters, value: number | string) => {
    setParameters((prev) => ({ ...prev, [name]: value }));
  };

  return (
    <AlgorithmLayout
      title="DBSCAN Clustering"
      description="Density-based clustering that can find arbitrarily shaped clusters and outliers"
      category="Machine Learning"
      difficulty="Intermediate"
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <Controls
            parameters={parameters}
            onParameterChange={handleParameterChange as (name: string, value: number | string) => void}
            onTrain={handleTrain}
            isTraining={trainMutation.isPending}
          />

          {result && (
            <Card>
              <CardHeader>
                <CardTitle>Results</CardTitle>
                <CardDescription>Clustering performance metrics</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Clusters Found</p>
                    <p className="text-2xl font-bold">{result.metrics.n_clusters}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Noise Points</p>
                    <p className="text-2xl font-bold">{result.metrics.n_noise}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Noise Ratio</p>
                    <p className="text-2xl font-bold">
                      {(result.metrics.noise_ratio * 100).toFixed(1)}%
                    </p>
                  </div>
                  {result.metrics.silhouette_score !== undefined && (
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">
                        Silhouette Score
                      </p>
                      <p className="text-2xl font-bold">
                        {result.metrics.silhouette_score.toFixed(3)}
                      </p>
                    </div>
                  )}
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">
                      Execution Time
                    </p>
                    <p className="text-2xl font-bold">
                      {result.execution_time_ms.toFixed(2)}ms
                    </p>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <h4 className="font-medium mb-2">Cluster Information</h4>
                  <div className="space-y-2">
                    {result.clusters
                      .filter((cluster) => !cluster.is_noise)
                      .map((cluster) => (
                        <div
                          key={cluster.cluster_id}
                          className="flex justify-between items-center text-sm"
                        >
                          <span className="font-medium">
                            Cluster {cluster.cluster_id}
                          </span>
                          <span className="text-muted-foreground">
                            {cluster.size} points
                          </span>
                        </div>
                      ))}
                    {result.clusters.find((c) => c.is_noise) && (
                      <div className="flex justify-between items-center text-sm border-t pt-2">
                        <span className="font-medium text-destructive">Noise</span>
                        <span className="text-muted-foreground">
                          {result.clusters.find((c) => c.is_noise)?.size} points
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        <div className="space-y-6">
          {trainMutation.isPending && (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <div className="text-center space-y-4">
                  <Loader2 className="h-8 w-8 animate-spin mx-auto text-primary" />
                  <p className="text-muted-foreground">Training DBSCAN model...</p>
                </div>
              </CardContent>
            </Card>
          )}

          {trainMutation.isError && (
            <Card className="border-destructive">
              <CardHeader>
                <CardTitle className="text-destructive">Training Failed</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-destructive">
                  {(trainMutation.error as Error)?.message ||
                    'An error occurred during training'}
                </p>
              </CardContent>
            </Card>
          )}

          {result && <Visualization data={result.visualization_data} />}
        </div>
      </div>

      <Documentation />
    </AlgorithmLayout>
  );
}

export default DBSCANClustering;
