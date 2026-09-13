import { ScatterPlot } from '@/components/visualizations/ScatterPlot';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/common/Card';

interface VisualizationProps {
  data: {
    clusters: Array<Array<{ x: number; y: number; cluster: number }>>;
    centroids: Array<{ x: number; y: number; cluster: number }>;
    n_clusters: number;
  };
  /** Centroid positions for the currently scrubbed-to iteration, if animating. Overrides data.centroids in the plot when provided. */
  currentCentroids?: number[][];
}

const clusterColors = [
  '#8884d8',
  '#82ca9d',
  '#ffc658',
  '#ff7c7c',
  '#a78bfa',
  '#fb923c',
  '#38bdf8',
  '#4ade80',
  '#f472b6',
  '#fb7185',
];

function Visualization({ data, currentCentroids }: VisualizationProps) {
  if (!data || !data.clusters || data.clusters.length === 0) {
    return null;
  }

  // Prepare data for ScatterPlot component
  const clusterNames = Array.from({ length: data.n_clusters }, (_, i) => `Cluster ${i}`);

  // Overlay the (possibly mid-convergence) centroids as an extra series so
  // you can watch them move across the fixed point cloud while scrubbing.
  const centroidPoints = (
    currentCentroids ?? data.centroids.map((c) => [c.x, c.y])
  ).map(([x, y]) => ({ x, y }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Cluster Visualization</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <ScatterPlot
          data={[...data.clusters, centroidPoints]}
          xKey="x"
          yKey="y"
          xLabel="Feature 1"
          yLabel="Feature 2"
          height={500}
          showGrid={true}
          showLegend={true}
          colors={[...clusterColors, '#4cf3ff']}
          clusterNames={[...clusterNames, 'Centroids']}
        />

        <div className="border-t pt-4">
          <h4 className="font-medium mb-2">Cluster Centers (Centroids)</h4>
          <div className="space-y-2">
            {centroidPoints.map((centroid, index) => (
              <div
                key={index}
                className="flex items-center justify-between text-sm"
              >
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: clusterColors[index % clusterColors.length] }}
                  />
                  <span className="font-medium">Cluster {index}</span>
                </div>
                <span className="text-muted-foreground font-mono">
                  ({centroid.x.toFixed(2)}, {centroid.y.toFixed(2)})
                </span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default Visualization;
