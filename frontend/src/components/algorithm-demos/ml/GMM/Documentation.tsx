import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';
import { Badge } from '@/components/ui/badge';

function Documentation() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Algorithm Overview</CardTitle>
          <CardDescription>Understanding Gaussian Mixture Models</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2">What is a Gaussian Mixture Model?</h4>
            <p className="text-sm text-muted-foreground">
              Gaussian Mixture Model (GMM) is a probabilistic model that assumes data points
              are generated from a mixture of a finite number of Gaussian distributions with
              unknown parameters. Unlike K-Means which performs hard clustering (each point
              belongs to exactly one cluster), GMM provides soft clustering where each point
              has a probability of belonging to each cluster.
            </p>
          </div>

          <div>
            <h4 className="font-semibold mb-2">How Does It Work?</h4>
            <p className="text-sm text-muted-foreground mb-2">
              GMM uses the Expectation-Maximization (EM) algorithm:
            </p>
            <ol className="list-decimal list-inside space-y-1 text-sm text-muted-foreground">
              <li>
                <strong>Initialization:</strong> Randomly initialize parameters (means,
                covariances, and weights) for K Gaussian components
              </li>
              <li>
                <strong>E-Step (Expectation):</strong> Calculate the probability that each
                data point belongs to each Gaussian component
              </li>
              <li>
                <strong>M-Step (Maximization):</strong> Update the parameters (means,
                covariances, weights) to maximize the likelihood given current
                probabilities
              </li>
              <li>
                <strong>Repeat:</strong> Iterate E and M steps until convergence
              </li>
            </ol>
          </div>

          <div>
            <h4 className="font-semibold mb-2">Key Parameters</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>
                <strong>Number of Components (K):</strong> The number of Gaussian
                distributions to fit. Must be chosen beforehand (similar to K-Means).
              </li>
              <li>
                <strong>Covariance Type:</strong> Determines the shape of clusters:
                <ul className="ml-4 mt-1 space-y-1">
                  <li>• <strong>Full:</strong> Each component has its own covariance matrix (most flexible)</li>
                  <li>• <strong>Tied:</strong> All components share the same covariance</li>
                  <li>• <strong>Diagonal:</strong> Covariance matrices are diagonal (axes-aligned ellipses)</li>
                  <li>• <strong>Spherical:</strong> Each component has a single variance (circular clusters)</li>
                </ul>
              </li>
              <li>
                <strong>Max Iterations:</strong> Maximum number of EM iterations before
                stopping. The algorithm may converge earlier.
              </li>
            </ul>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Complexity Analysis</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2">Time Complexity</h4>
            <Badge variant="secondary" className="font-mono">
              O(n × k × d² × i)
            </Badge>
            <p className="text-sm text-muted-foreground mt-2">
              Where n = number of samples, k = number of components, d = number of
              features, and i = number of iterations. The d² factor comes from computing
              covariance matrices.
            </p>
          </div>

          <div>
            <h4 className="font-semibold mb-2">Space Complexity</h4>
            <Badge variant="secondary" className="font-mono">
              O(k × d²)
            </Badge>
            <p className="text-sm text-muted-foreground mt-2">
              Storage for k covariance matrices (d × d each), plus means and weights. For
              'spherical' or 'diag' covariance, space complexity reduces to O(k × d).
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Advantages</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex gap-2">
                <span className="text-green-500">✓</span>
                <span>Provides probabilistic cluster assignments</span>
              </li>
              <li className="flex gap-2">
                <span className="text-green-500">✓</span>
                <span>Can model elliptical clusters with different shapes and orientations</span>
              </li>
              <li className="flex gap-2">
                <span className="text-green-500">✓</span>
                <span>Naturally handles uncertainty in cluster membership</span>
              </li>
              <li className="flex gap-2">
                <span className="text-green-500">✓</span>
                <span>Can be used for density estimation and anomaly detection</span>
              </li>
              <li className="flex gap-2">
                <span className="text-green-500">✓</span>
                <span>Flexible covariance structures for different data patterns</span>
              </li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Limitations</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex gap-2">
                <span className="text-red-500">✗</span>
                <span>Computationally expensive for large datasets</span>
              </li>
              <li className="flex gap-2">
                <span className="text-red-500">✗</span>
                <span>Sensitive to initialization (may converge to local optima)</span>
              </li>
              <li className="flex gap-2">
                <span className="text-red-500">✗</span>
                <span>Requires specification of number of components</span>
              </li>
              <li className="flex gap-2">
                <span className="text-red-500">✗</span>
                <span>Assumes Gaussian distribution of clusters</span>
              </li>
              <li className="flex gap-2">
                <span className="text-red-500">✗</span>
                <span>Can be unstable with insufficient data per component</span>
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Common Use Cases</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <h4 className="font-semibold text-sm mb-2">Density Estimation</h4>
              <p className="text-sm text-muted-foreground">
                Modeling the probability distribution of data for understanding data
                structure and generating synthetic samples.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-sm mb-2">Anomaly Detection</h4>
              <p className="text-sm text-muted-foreground">
                Identifying outliers based on low probability under the learned mixture
                model.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-sm mb-2">Image Segmentation</h4>
              <p className="text-sm text-muted-foreground">
                Separating image regions by modeling pixel color distributions as mixtures
                of Gaussians.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-sm mb-2">Customer Segmentation</h4>
              <p className="text-sm text-muted-foreground">
                Grouping customers with soft cluster memberships, allowing for nuanced
                targeting strategies.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-sm mb-2">Speech Recognition</h4>
              <p className="text-sm text-muted-foreground">
                Modeling acoustic features of phonemes as Gaussian mixtures in hidden
                Markov models.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Evaluation Metrics</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold text-sm mb-1">Silhouette Score</h4>
            <p className="text-sm text-muted-foreground">
              Measures how similar a point is to its own cluster compared to other
              clusters. Range: [-1, 1], where higher is better.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-sm mb-1">Davies-Bouldin Index</h4>
            <p className="text-sm text-muted-foreground">
              Ratio of within-cluster to between-cluster distances. Lower values indicate
              better clustering.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-sm mb-1">BIC (Bayesian Information Criterion)</h4>
            <p className="text-sm text-muted-foreground">
              Model selection criterion that penalizes complexity. Lower BIC indicates a
              better model fit with appropriate complexity.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-sm mb-1">AIC (Akaike Information Criterion)</h4>
            <p className="text-sm text-muted-foreground">
              Similar to BIC but with different penalty. Useful for comparing models with
              different numbers of components.
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Related Algorithms</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">K-Means Clustering</Badge>
            <Badge variant="outline">DBSCAN</Badge>
            <Badge variant="outline">Hierarchical Clustering</Badge>
            <Badge variant="outline">Expectation-Maximization</Badge>
            <Badge variant="outline">Hidden Markov Models</Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default Documentation;
