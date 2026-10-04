import { Card, CardContent, CardHeader, CardTitle } from '@/components/common/Card';
import { BookOpen, CheckCircle, XCircle, Lightbulb } from 'lucide-react';

function Documentation() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            Algorithm Theory
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based
            clustering algorithm that groups together points that are closely packed together, while
            marking points in low-density regions as outliers. It works by:
          </p>
          <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
            <li>Defining core points as those with at least min_samples neighbors within eps distance</li>
            <li>Forming clusters by connecting core points that are within eps of each other</li>
            <li>Assigning border points (non-core points within eps of a core point) to clusters</li>
            <li>Marking remaining points as noise (outliers)</li>
          </ol>
          <p className="text-sm text-muted-foreground">
            Unlike K-Means, DBSCAN can discover clusters of arbitrary shape and automatically determines
            the number of clusters based on data density.
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              Advantages
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="text-green-500 mt-0.5">•</span>
                <span>Discovers arbitrarily shaped clusters</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500 mt-0.5">•</span>
                <span>Identifies and handles noise/outliers</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500 mt-0.5">•</span>
                <span>Doesn't require specifying number of clusters</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500 mt-0.5">•</span>
                <span>Robust to outliers</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500 mt-0.5">•</span>
                <span>Works well with non-convex clusters</span>
              </li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <XCircle className="h-5 w-5 text-red-500" />
              Limitations
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="text-red-500 mt-0.5">•</span>
                <span>Sensitive to eps and min_samples parameters</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-500 mt-0.5">•</span>
                <span>Struggles with varying density clusters</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-500 mt-0.5">•</span>
                <span>Not suitable for high-dimensional data</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-500 mt-0.5">•</span>
                <span>Cannot predict cluster for new data points</span>
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            Use Cases
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <h4 className="font-medium text-sm">Anomaly Detection</h4>
              <p className="text-xs text-muted-foreground">
                Identify outliers and unusual patterns in data by marking low-density points as noise
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium text-sm">Spatial Data Clustering</h4>
              <p className="text-xs text-muted-foreground">
                Cluster geographic locations, such as identifying crime hotspots or disease outbreak zones
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium text-sm">Pattern Recognition</h4>
              <p className="text-xs text-muted-foreground">
                Find complex patterns in image segmentation, sensor data analysis, and network traffic analysis
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Complexity Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <h4 className="font-medium text-sm mb-2">Time Complexity</h4>
              <p className="text-sm font-mono bg-muted px-3 py-2 rounded">O(n*log(n))</p>
              <p className="text-xs text-muted-foreground mt-2">
                With spatial indexing structures like KD-trees or R-trees. Without indexing: O(n²)
              </p>
            </div>
            <div>
              <h4 className="font-medium text-sm mb-2">Space Complexity</h4>
              <p className="text-sm font-mono bg-muted px-3 py-2 rounded">O(n)</p>
              <p className="text-xs text-muted-foreground mt-2">
                Stores cluster labels and visited status for each point
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Parameter Guide</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-medium text-sm mb-2">Epsilon (eps)</h4>
            <p className="text-xs text-muted-foreground">
              Maximum distance between two samples for one to be considered in the neighborhood of the other.
              Smaller values create tighter, more granular clusters. Larger values merge clusters together.
              Use the K-distance graph method to find optimal eps.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-sm mb-2">Minimum Samples (min_samples)</h4>
            <p className="text-xs text-muted-foreground">
              Minimum number of samples in a neighborhood for a point to be considered a core point.
              Higher values create denser, more robust clusters but may classify more points as noise.
              A good rule of thumb is to set this to at least the number of dimensions plus one.
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Metrics Explanation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-medium text-sm mb-2">Number of Clusters</h4>
            <p className="text-xs text-muted-foreground">
              Total number of distinct clusters found by DBSCAN. Unlike K-Means, this is determined
              automatically based on data density and the eps/min_samples parameters.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-sm mb-2">Noise Points</h4>
            <p className="text-xs text-muted-foreground">
              Points that don't belong to any cluster because they're in low-density regions.
              These are potential outliers or anomalies in your data.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-sm mb-2">Silhouette Score</h4>
            <p className="text-xs text-muted-foreground">
              Measures how similar a point is to its own cluster compared to other clusters.
              Ranges from -1 to 1, where higher values indicate better-defined clusters. Only
              calculated when at least 2 clusters are found (excluding noise).
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default Documentation;
