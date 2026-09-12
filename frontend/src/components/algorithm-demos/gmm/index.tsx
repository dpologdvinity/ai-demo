import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiService } from '@/services/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';
import Button from '@/components/common/Button';
import Controls from './Controls';
import Visualization from './Visualization';
import Documentation from './Documentation';
import { Loader2 } from 'lucide-react';

interface GMMParameters {
  n_components: number;
  covariance_type: 'full' | 'tied' | 'diag' | 'spherical';
  max_iter: number;
  n_samples: number;
  random_state: number;
}

interface GMMResult {
  metrics: {
    silhouette_score: number;
    davies_bouldin_score: number;
    bic: number;
    aic: number;
    log_likelihood: number;
  };
  predictions: number[];
  probabilities: number[][];
  visualization_data: {
    x: number[];
    y: number[];
    labels: number[];
    contour: {
      x_grid: number[];
      y_grid: number[];
      densities: number[][];
    };
    means: number[][];
    true_labels: number[];
  };
  parameters: {
    means: number[][];
    covariances: any;
    weights: number[];
    n_components: number;
    covariance_type: string;
  };
  execution_time_ms: number;
  model_info: {
    n_components: number;
    covariance_type: string;
    max_iter: number;
    converged: boolean;
    n_iter: number;
    n_features: number;
  };
}

function GaussianMixtureModel() {
  const [parameters, setParameters] = useState<GMMParameters>({
    n_components: 3,
    covariance_type: 'full',
    max_iter: 100,
    n_samples: 300,
    random_state: 42,
  });

  const [result, setResult] = useState<GMMResult | null>(null);

  const trainMutation = useMutation({
    mutationFn: (params: GMMParameters) =>
      apiService.post<GMMResult>('/api/ml/gmm/train', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (
    name: keyof GMMParameters,
    value: number | string
  ) => {
    setParameters((prev) => ({ ...prev, [name]: value }));
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold mb-2">Gaussian Mixture Model</h2>
        <p className="text-muted-foreground">
          Probabilistic clustering assuming data is a mixture of Gaussians
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
                <CardDescription>Clustering quality metrics</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">
                      Silhouette Score
                    </p>
                    <p className="text-2xl font-bold">
                      {result.metrics.silhouette_score.toFixed(3)}
                    </p>
                    <p className="text-xs text-muted-foreground">Higher is better</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">
                      Davies-Bouldin
                    </p>
                    <p className="text-2xl font-bold">
                      {result.metrics.davies_bouldin_score.toFixed(3)}
                    </p>
                    <p className="text-xs text-muted-foreground">Lower is better</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">BIC</p>
                    <p className="text-2xl font-bold">
                      {result.metrics.bic.toFixed(1)}
                    </p>
                    <p className="text-xs text-muted-foreground">Lower is better</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">AIC</p>
                    <p className="text-2xl font-bold">
                      {result.metrics.aic.toFixed(1)}
                    </p>
                    <p className="text-xs text-muted-foreground">Lower is better</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">
                      Log-Likelihood
                    </p>
                    <p className="text-2xl font-bold">
                      {result.metrics.log_likelihood.toFixed(2)}
                    </p>
                  </div>
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
                  <h4 className="font-medium mb-2">Model Information</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Converged:</span>
                      <span className="font-medium">
                        {result.model_info.converged ? 'Yes' : 'No'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Iterations:</span>
                      <span className="font-medium">{result.model_info.n_iter}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Components:</span>
                      <span className="font-medium">
                        {result.model_info.n_components}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Covariance Type:</span>
                      <span className="font-medium">
                        {result.model_info.covariance_type}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <h4 className="font-medium mb-2">Component Weights</h4>
                  <div className="space-y-2">
                    {result.parameters.weights.map((weight, idx) => (
                      <div
                        key={idx}
                        className="flex justify-between items-center text-sm"
                      >
                        <span className="font-medium">Component {idx}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-32 bg-secondary rounded-full h-2">
                            <div
                              className="bg-primary h-2 rounded-full"
                              style={{ width: `${weight * 100}%` }}
                            />
                          </div>
                          <span className="text-muted-foreground w-12 text-right">
                            {(weight * 100).toFixed(1)}%
                          </span>
                        </div>
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
                  <p className="text-muted-foreground">
                    Training Gaussian Mixture Model...
                  </p>
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

export default GaussianMixtureModel;
