import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ZAxis,
} from 'recharts';

interface VisualizationData {
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
}

interface VisualizationProps {
  data: VisualizationData;
}

const COLORS = [
  '#8b5cf6', // Purple
  '#06b6d4', // Cyan
  '#f59e0b', // Amber
  '#10b981', // Green
  '#ef4444', // Red
  '#3b82f6', // Blue
  '#ec4899', // Pink
  '#14b8a6', // Teal
  '#f97316', // Orange
  '#6366f1', // Indigo
];

function Visualization({ data }: VisualizationProps) {
  // Prepare scatter data by cluster
  const clusterData: { [key: number]: Array<{ x: number; y: number }> } = {};
  const centerData: Array<{ x: number; y: number; cluster: number }> = [];

  // Group points by cluster
  data.x.forEach((x, idx) => {
    const y = data.y[idx];
    const cluster = data.labels[idx];

    if (!clusterData[cluster]) {
      clusterData[cluster] = [];
    }
    clusterData[cluster].push({ x, y });
  });

  // Add cluster centers
  data.means.forEach((mean, idx) => {
    centerData.push({
      x: mean[0],
      y: mean[1],
      cluster: idx,
    });
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Clustering Visualization</CardTitle>
        <CardDescription>
          Data points colored by predicted cluster with Gaussian centers shown
        </CardDescription>
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
            <XAxis type="number" dataKey="x" name="X" />
            <YAxis type="number" dataKey="y" name="Y" />
            <ZAxis type="number" range={[50, 50]} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
                      <p className="text-sm font-medium">
                        Cluster {payload[0].name.split(' ')[1]}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        X: {data.x.toFixed(2)}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Y: {data.y.toFixed(2)}
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend />

            {/* Plot each cluster */}
            {Object.entries(clusterData).map(([cluster, points]) => (
              <Scatter
                key={`cluster-${cluster}`}
                name={`Cluster ${cluster}`}
                data={points}
                fill={COLORS[parseInt(cluster) % COLORS.length]}
                fillOpacity={0.6}
              />
            ))}

            {/* Plot cluster centers */}
            <Scatter
              name="Cluster Centers"
              data={centerData}
              fill="#000000"
              shape="cross"
              line={false}
            />
          </ScatterChart>
        </ResponsiveContainer>

        <div className="mt-4 text-sm text-muted-foreground">
          <p>
            <strong>Interpretation:</strong> Each color represents a predicted Gaussian
            component. Black crosses mark the learned component means. The GMM assigns
            soft probabilities to each point for belonging to each component.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

export default Visualization;
