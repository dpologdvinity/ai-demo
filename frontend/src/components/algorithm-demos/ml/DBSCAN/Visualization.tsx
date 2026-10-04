import { ScatterPlot } from '@/components/visualizations/ScatterPlot';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/common/Card';

interface VisualizationProps {
  data: {
    clusters: Array<Array<{ x: number; y: number; cluster: number }>>;
    noise_points: Array<{ x: number; y: number; cluster: number }>;
    n_clusters: number;
  };
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

const noiseColor = '#94a3b8'; // Gray color for noise points

function Visualization({ data }: VisualizationProps) {
  if (!data || !data.clusters) {
    return null;
  }

  // Prepare data for ScatterPlot - include noise points as a separate cluster
  const allClusters = [...data.clusters];
  if (data.noise_points && data.noise_points.length > 0) {
    allClusters.push(data.noise_points);
  }

  // Create cluster names including noise
  const clusterNames = [
    ...Array.from({ length: data.n_clusters }, (_, i) => `Cluster ${i}`),
    ...(data.noise_points && data.noise_points.length > 0 ? ['Noise'] : []),
  ];

  // Create color array including noise color
  const colors = [
    ...clusterColors.slice(0, data.n_clusters),
    ...(data.noise_points && data.noise_points.length > 0 ? [noiseColor] : []),
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Cluster Visualization</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <ScatterPlot
          data={allClusters}
          xKey="x"
          yKey="y"
          xLabel="Feature 1"
          yLabel="Feature 2"
          height={500}
          showGrid={true}
          showLegend={true}
          colors={colors}
          clusterNames={clusterNames}
        />

        <div className="border-t pt-4">
          <h4 className="font-medium mb-2">Cluster Summary</h4>
          <div className="space-y-2">
            {data.clusters.map((cluster, index) => {
              if (cluster.length === 0) return null;
              return (
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
                  <span className="text-muted-foreground">
                    {cluster.length} points
                  </span>
                </div>
              );
            })}
            {data.noise_points && data.noise_points.length > 0 && (
              <div className="flex items-center justify-between text-sm border-t pt-2">
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: noiseColor }}
                  />
                  <span className="font-medium text-muted-foreground">Noise Points</span>
                </div>
                <span className="text-muted-foreground">
                  {data.noise_points.length} points
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="border-t pt-4 text-sm text-muted-foreground">
          <p>
            DBSCAN identified {data.n_clusters} cluster{data.n_clusters !== 1 ? 's' : ''}
            {data.noise_points && data.noise_points.length > 0 &&
              ` and ${data.noise_points.length} noise point${data.noise_points.length !== 1 ? 's' : ''}`}.
            Noise points (shown in gray) are outliers that don't belong to any cluster.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

export default Visualization;
