import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  if (!algorithmInfo) {
    return null;
  }

  const metadata = algorithmInfo.metadata || algorithmInfo;

  return (
    <div className="space-y-6">
      {/* Overview */}
      <Card>
        <CardHeader>
          <CardTitle>Algorithm Overview</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h3 className="font-semibold mb-2">What is Hierarchical Clustering?</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {metadata.theory ||
                'Hierarchical clustering is an unsupervised learning algorithm that builds a hierarchy of clusters. ' +
                'It works by either iteratively merging smaller clusters into larger ones (agglomerative) or ' +
                'splitting larger clusters into smaller ones (divisive). The result is a tree-like structure ' +
                'called a dendrogram that shows the hierarchical relationships between clusters.'}
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">Difficulty: {metadata.difficulty}</Badge>
            <Badge variant="outline">Category: {metadata.category}</Badge>
            {metadata.tags?.map((tag: string) => (
              <Badge key={tag} variant="secondary">
                {tag}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* How It Works */}
      <Card>
        <CardHeader>
          <CardTitle>How It Works</CardTitle>
        </CardHeader>
        <CardContent>
          <ol className="list-decimal list-inside space-y-3 text-sm text-muted-foreground">
            <li>
              <strong>Initialize:</strong> Start with each data point as its own cluster (N clusters for N points)
            </li>
            <li>
              <strong>Compute Distances:</strong> Calculate pairwise distances between all clusters using the chosen metric
            </li>
            <li>
              <strong>Merge Closest Clusters:</strong> Find the two closest clusters and merge them into one
            </li>
            <li>
              <strong>Update Distances:</strong> Recalculate distances using the linkage criterion (ward, complete, average, or single)
            </li>
            <li>
              <strong>Repeat:</strong> Continue merging until only the desired number of clusters remain
            </li>
            <li>
              <strong>Build Dendrogram:</strong> Record all merge operations to create a hierarchical tree structure
            </li>
          </ol>
        </CardContent>
      </Card>

      {/* Linkage Methods */}
      <Card>
        <CardHeader>
          <CardTitle>Linkage Methods</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 text-sm">
            <div>
              <h4 className="font-semibold text-foreground">Ward Linkage</h4>
              <p className="text-muted-foreground">
                Minimizes the variance within clusters. Only works with Euclidean distance.
                Tends to create compact, spherical clusters of similar size.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-foreground">Complete Linkage</h4>
              <p className="text-muted-foreground">
                Uses the maximum distance between any two points in different clusters.
                Tends to create compact clusters and is less sensitive to outliers.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-foreground">Average Linkage</h4>
              <p className="text-muted-foreground">
                Uses the average distance between all pairs of points in different clusters.
                A compromise between single and complete linkage.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-foreground">Single Linkage</h4>
              <p className="text-muted-foreground">
                Uses the minimum distance between any two points in different clusters.
                Can create elongated clusters but is sensitive to outliers (chaining effect).
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Complexity Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Complexity Analysis</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Time Complexity</span>
            <code className="px-2 py-1 bg-secondary rounded text-sm font-mono">
              {metadata.complexity?.time || 'O(n³)'}
            </code>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Space Complexity</span>
            <code className="px-2 py-1 bg-secondary rounded text-sm font-mono">
              {metadata.complexity?.space || 'O(n²)'}
            </code>
          </div>
          <p className="text-xs text-muted-foreground pt-2">
            The high complexity makes hierarchical clustering unsuitable for large datasets (typically n {'>'} 1000).
            For larger datasets, consider using K-Means or DBSCAN instead.
          </p>
        </CardContent>
      </Card>

      {/* Pros and Cons */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-green-600 dark:text-green-400">Advantages</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc list-inside space-y-2 text-sm text-muted-foreground">
              {metadata.pros?.map((pro: string, index: number) => (
                <li key={index}>{pro}</li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-red-600 dark:text-red-400">Limitations</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc list-inside space-y-2 text-sm text-muted-foreground">
              {metadata.cons?.map((con: string, index: number) => (
                <li key={index}>{con}</li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>

      {/* Use Cases */}
      <Card>
        <CardHeader>
          <CardTitle>Common Use Cases</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc list-inside space-y-2 text-sm text-muted-foreground">
            {metadata.use_cases?.map((useCase: string, index: number) => (
              <li key={index}>{useCase}</li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* Related Algorithms */}
      {metadata.related_algorithms && metadata.related_algorithms.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Related Algorithms</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {metadata.related_algorithms.map((algo: string) => (
                <Badge key={algo} variant="outline" className="capitalize">
                  {algo.replace('-', ' ')}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
