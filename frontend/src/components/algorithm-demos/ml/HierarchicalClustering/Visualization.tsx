import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useMemo } from 'react';

interface VisualizationProps {
  result: {
    visualization_data: {
      scatter: {
        X: number[][];
        labels: number[];
        n_clusters: number;
      };
      dendrogram: {
        linkage_matrix: number[][];
        n_clusters: number;
        linkage_method: string;
        affinity: string;
      };
    };
    model_info: {
      cluster_sizes: Record<string, number>;
    };
  };
}

// Color palette for clusters
const CLUSTER_COLORS = [
  '#3b82f6', // blue
  '#ef4444', // red
  '#10b981', // green
  '#f59e0b', // amber
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#14b8a6', // teal
  '#f97316', // orange
  '#6366f1', // indigo
  '#84cc16', // lime
];

export function Visualization({ result }: VisualizationProps) {
  const { scatter, dendrogram } = result.visualization_data;

  // Prepare scatter plot data
  const scatterData = useMemo(() => {
    return scatter.X.map((point, idx) => ({
      x: point[0],
      y: point[1],
      cluster: scatter.labels[idx],
    }));
  }, [scatter]);

  // Group data by cluster for rendering
  const clusterGroups = useMemo(() => {
    const groups: Record<number, Array<{ x: number; y: number }>> = {};
    scatterData.forEach((point) => {
      if (!groups[point.cluster]) {
        groups[point.cluster] = [];
      }
      groups[point.cluster].push({ x: point.x, y: point.y });
    });
    return groups;
  }, [scatterData]);

  // Prepare dendrogram visualization (simplified representation)
  const dendrogramHeight = useMemo(() => {
    if (!dendrogram.linkage_matrix || dendrogram.linkage_matrix.length === 0) {
      return 0;
    }
    // The last row contains the highest merge distance
    const lastMerge = dendrogram.linkage_matrix[dendrogram.linkage_matrix.length - 1];
    return lastMerge[2];
  }, [dendrogram]);

  return (
    <div className="space-y-6">
      {/* Scatter Plot with Clusters */}
      <Card>
        <CardHeader>
          <CardTitle>Cluster Assignments</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart
              margin={{
                top: 20,
                right: 20,
                bottom: 20,
                left: 20,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="Feature 1" />
              <YAxis type="number" dataKey="y" name="Feature 2" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              {Object.entries(clusterGroups).map(([cluster, points]) => (
                <Scatter
                  key={cluster}
                  name={`Cluster ${cluster}`}
                  data={points}
                  fill={CLUSTER_COLORS[parseInt(cluster) % CLUSTER_COLORS.length]}
                  shape="circle"
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Dendrogram Info and Cluster Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Dendrogram Info</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Linkage Method</span>
              <span className="font-mono font-semibold capitalize">
                {dendrogram.linkage_method}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Distance Metric</span>
              <span className="font-mono font-semibold capitalize">
                {dendrogram.affinity}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Number of Merges</span>
              <span className="font-mono font-semibold">
                {dendrogram.linkage_matrix.length}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Max Merge Distance</span>
              <span className="font-mono font-semibold">
                {dendrogramHeight.toFixed(2)}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Cluster Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {Object.entries(result.model_info.cluster_sizes).map(([cluster, size]) => (
                <div key={cluster} className="flex items-center gap-3">
                  <div
                    className="w-4 h-4 rounded-full"
                    style={{
                      backgroundColor: CLUSTER_COLORS[parseInt(cluster) % CLUSTER_COLORS.length],
                    }}
                  />
                  <span className="text-sm text-muted-foreground">Cluster {cluster}</span>
                  <span className="font-mono font-semibold ml-auto">{size} samples</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Dendrogram Visualization Note */}
      <Card className="bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
        <CardContent className="pt-6">
          <p className="text-sm text-blue-900 dark:text-blue-100">
            <strong>Note:</strong> Hierarchical clustering builds a tree structure (dendrogram) showing how clusters are merged.
            The linkage matrix contains {dendrogram.linkage_matrix.length} merge operations, with the final merge at distance {dendrogramHeight.toFixed(2)}.
            Interactive dendrogram visualization can be added using libraries like d3-hierarchy.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
