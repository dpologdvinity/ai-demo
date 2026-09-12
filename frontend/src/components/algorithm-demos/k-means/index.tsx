import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiService } from '@/services/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';
import Button from '@/components/common/Button';
import Controls from './Controls';
import Visualization from './Visualization';
import Documentation from './Documentation';
import { Loader2 } from 'lucide-react';

interface KMeansParameters {
  n_clusters: number;
  max_iter: number;
  n_init: number;
  random_state: number;
  n_samples: number;
}

interface ClusterInfo {
  cluster_id: number;
  center: number[];
  size: number;
}

interface KMeansResult {
  success: boolean;
  metrics: {
    inertia: number;
    silhouette_score?: number;
  };
  clusters: ClusterInfo[];
  visualization_data: {
    clusters: Array<Array<{ x: number; y: number; cluster: number }>>;
    centroids: Array<{ x: number; y: number; cluster: number }>;
    n_clusters: number;
  };
  execution_time_ms: number;
  parameters_used: KMeansParameters;
  n_iterations?: number;
  error?: string;
}

function KMeansClustering() {
  const [parameters, setParameters] = useState<KMeansParameters>({
    n_clusters: 3,
    max_iter: 300,
    n_init: 10,
    random_state: 42,
    n_samples: 300,
  });

  const [result, setResult] = useState<KMeansResult | null>(null);

  const trainMutation = useMutation({
    mutationFn: (params: KMeansParameters) =>
      apiService.post<KMeansResult>('/api/ml/k-means/train', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof KMeansParameters, value: number) => {
    setParameters((prev) => ({ ...prev, [name]: value }));
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold mb-2">K-Means Clustering</h2>
        <p className="text-muted-foreground">
          Unsupervised learning algorithm that groups data into K clusters
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <Controls
            parameters={parameters}
            onParameterChange={handleParameterChange}
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
                    <p className="text-sm font-medium text-muted-foreground">Inertia</p>
                    <p className="text-2xl font-bold">
                      {result.metrics.inertia.toFixed(2)}
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
                  {result.n_iterations !== undefined && (
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">
                        Iterations
                      </p>
                      <p className="text-2xl font-bold">{result.n_iterations}</p>
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
                    {result.clusters.map((cluster) => (
                      <div
                        key={cluster.cluster_id}
                        className="flex justify-between items-center text-sm"
                      >
                        <span className="font-medium">Cluster {cluster.cluster_id}</span>
                        <span className="text-muted-foreground">
                          {cluster.size} points
                        </span>
                      </div>
                    ))}
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
                  <p className="text-muted-foreground">Training K-Means model...</p>
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
    </div>
  );
}

export default KMeansClustering;
