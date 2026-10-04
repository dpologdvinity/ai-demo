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
            K-Means is an iterative algorithm that divides data into K clusters. It works by:
          </p>
          <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
            <li>Randomly initializing K centroids</li>
            <li>Assigning each point to the nearest centroid</li>
            <li>Updating centroids to the mean of assigned points</li>
            <li>Repeating steps 2-3 until convergence</li>
          </ol>
          <p className="text-sm text-muted-foreground">
            The algorithm converges when centroid positions no longer change significantly
            or the maximum number of iterations is reached.
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
                <span>Simple and intuitive algorithm</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500 mt-0.5">•</span>
                <span>Fast and efficient for large datasets</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500 mt-0.5">•</span>
                <span>Works well with spherical clusters</span>
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
                <span>Requires specifying K in advance</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-500 mt-0.5">•</span>
                <span>Sensitive to initial centroid placement</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-500 mt-0.5">•</span>
                <span>Struggles with non-spherical clusters</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-500 mt-0.5">•</span>
                <span>Affected by outliers</span>
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
              <h4 className="font-medium text-sm">Customer Segmentation</h4>
              <p className="text-xs text-muted-foreground">
                Group customers based on purchasing behavior, demographics, or engagement patterns
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium text-sm">Image Compression</h4>
              <p className="text-xs text-muted-foreground">
                Reduce image file sizes by clustering similar colors together
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium text-sm">Document Clustering</h4>
              <p className="text-xs text-muted-foreground">
                Organize large document collections by topic or content similarity
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
              <p className="text-sm font-mono bg-muted px-3 py-2 rounded">O(n*k*i)</p>
              <p className="text-xs text-muted-foreground mt-2">
                Where n = number of samples, k = number of clusters, i = number of iterations
              </p>
            </div>
            <div>
              <h4 className="font-medium text-sm mb-2">Space Complexity</h4>
              <p className="text-sm font-mono bg-muted px-3 py-2 rounded">O(n*k)</p>
              <p className="text-xs text-muted-foreground mt-2">
                Stores cluster assignments and centroid positions
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Metrics Explanation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-medium text-sm mb-2">Inertia</h4>
            <p className="text-xs text-muted-foreground">
              Sum of squared distances of samples to their closest cluster center. Lower values
              indicate tighter, more cohesive clusters. However, inertia decreases as K increases,
              so it should be used in conjunction with other metrics.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-sm mb-2">Silhouette Score</h4>
            <p className="text-xs text-muted-foreground">
              Measures how similar a point is to its own cluster compared to other clusters.
              Ranges from -1 to 1, where higher values indicate better-defined clusters. Values
              near 0 indicate overlapping clusters.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default Documentation;
