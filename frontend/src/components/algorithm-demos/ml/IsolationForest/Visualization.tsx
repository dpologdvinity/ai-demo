import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface VisualizationData {
  normal_points: Array<{
    x: number;
    y: number;
    score: number;
    prediction: number;
    true_label?: number;
  }>;
  anomaly_points: Array<{
    x: number;
    y: number;
    score: number;
    prediction: number;
    true_label?: number;
  }>;
  anomaly_scores: Array<{
    index: number;
    score: number;
    prediction: number;
  }>;
  score_stats: {
    min: number;
    max: number;
    mean: number;
    median: number;
    std: number;
  };
  n_normal: number;
  n_anomalies: number;
}

interface TrainingResult {
  visualization_data: VisualizationData;
  metrics: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

export function Visualization({ result }: VisualizationProps) {
  const { visualization_data } = result;

  // Prepare score distribution data (binned histogram)
  const prepareScoreDistribution = () => {
    const scores = visualization_data.anomaly_scores.map((s) => s.score);
    const bins = 30;
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const binSize = (max - min) / bins;

    const histogram: Array<{ bin: string; count: number; isAnomalous: boolean }> = [];

    for (let i = 0; i < bins; i++) {
      const binStart = min + i * binSize;
      const binEnd = binStart + binSize;
      const binCenter = (binStart + binEnd) / 2;

      const count = scores.filter((s) => s >= binStart && s < binEnd).length;

      // Consider scores in the lower 30% as anomalous for coloring
      const threshold = min + (max - min) * 0.3;
      const isAnomalous = binCenter < threshold;

      histogram.push({
        bin: binCenter.toFixed(3),
        count,
        isAnomalous,
      });
    }

    return histogram;
  };

  const scoreDistribution = prepareScoreDistribution();

  // Prepare confusion matrix data if available
  const confusionMatrix = result.metrics.confusion_matrix;

  return (
    <div className="space-y-4">
      <Tabs defaultValue="scatter" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="scatter">Scatter Plot</TabsTrigger>
          <TabsTrigger value="scores">Anomaly Scores</TabsTrigger>
          {confusionMatrix && <TabsTrigger value="confusion">Confusion Matrix</TabsTrigger>}
        </TabsList>

        {/* Scatter Plot Tab */}
        <TabsContent value="scatter">
          <Card>
            <CardHeader>
              <CardTitle>Anomaly Detection Visualization</CardTitle>
              <p className="text-sm text-muted-foreground">
                Normal points (blue) vs Anomalies (red). Size represents anomaly score magnitude.
              </p>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={500}>
                <ScatterChart
                  margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name="Feature 1"
                    label={{ value: 'Feature 1', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name="Feature 2"
                    label={{ value: 'Feature 2', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-background border rounded p-2 shadow-lg">
                            <p className="font-semibold">
                              {data.prediction === 1 ? 'Normal' : 'Anomaly'}
                            </p>
                            <p className="text-sm">X: {data.x.toFixed(3)}</p>
                            <p className="text-sm">Y: {data.y.toFixed(3)}</p>
                            <p className="text-sm">Score: {data.score.toFixed(4)}</p>
                            {data.true_label !== undefined && (
                              <p className="text-sm text-muted-foreground">
                                True: {data.true_label === 1 ? 'Normal' : 'Anomaly'}
                              </p>
                            )}
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Legend />
                  <Scatter
                    name="Normal Points"
                    data={visualization_data.normal_points}
                    fill="#3b82f6"
                    opacity={0.6}
                  />
                  <Scatter
                    name="Anomalies"
                    data={visualization_data.anomaly_points}
                    fill="#ef4444"
                    opacity={0.8}
                  />
                </ScatterChart>
              </ResponsiveContainer>

              <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-blue-500 rounded-full"></div>
                  <span>Normal: {visualization_data.n_normal} points</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-red-500 rounded-full"></div>
                  <span>Anomaly: {visualization_data.n_anomalies} points</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Anomaly Scores Tab */}
        <TabsContent value="scores">
          <Card>
            <CardHeader>
              <CardTitle>Anomaly Score Distribution</CardTitle>
              <p className="text-sm text-muted-foreground">
                Distribution of anomaly scores. More negative scores indicate higher anomaly likelihood.
              </p>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={scoreDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="bin"
                    label={{ value: 'Anomaly Score', position: 'insideBottom', offset: -10 }}
                    tick={{ fontSize: 10 }}
                    interval={4}
                  />
                  <YAxis
                    label={{ value: 'Count', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-background border rounded p-2 shadow-lg">
                            <p className="text-sm">Score: {data.bin}</p>
                            <p className="text-sm font-semibold">Count: {data.count}</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar dataKey="count">
                    {scoreDistribution.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.isAnomalous ? '#ef4444' : '#3b82f6'}
                        opacity={0.7}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              <div className="mt-4 p-4 bg-muted rounded-lg">
                <h4 className="font-semibold mb-2">Score Statistics</h4>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-muted-foreground">Mean</p>
                    <p className="font-mono">{visualization_data.score_stats.mean.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Median</p>
                    <p className="font-mono">{visualization_data.score_stats.median.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Min (Most Anomalous)</p>
                    <p className="font-mono text-red-600">{visualization_data.score_stats.min.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Max (Most Normal)</p>
                    <p className="font-mono text-blue-600">{visualization_data.score_stats.max.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Std Dev</p>
                    <p className="font-mono">{visualization_data.score_stats.std.toFixed(4)}</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Confusion Matrix Tab */}
        {confusionMatrix && (
          <TabsContent value="confusion">
            <Card>
              <CardHeader>
                <CardTitle>Confusion Matrix</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Model predictions vs ground truth labels
                </p>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col items-center">
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div></div>
                    <div className="font-semibold text-sm">Predicted Normal</div>
                    <div className="font-semibold text-sm">Predicted Anomaly</div>

                    <div className="font-semibold text-sm self-center">True Normal</div>
                    <div className="bg-blue-100 border-2 border-blue-300 p-6 rounded-lg">
                      <p className="text-3xl font-bold">{confusionMatrix[0][0]}</p>
                      <p className="text-xs text-muted-foreground">True Negatives</p>
                    </div>
                    <div className="bg-red-100 border-2 border-red-300 p-6 rounded-lg">
                      <p className="text-3xl font-bold">{confusionMatrix[0][1]}</p>
                      <p className="text-xs text-muted-foreground">False Positives</p>
                    </div>

                    <div className="font-semibold text-sm self-center">True Anomaly</div>
                    <div className="bg-red-100 border-2 border-red-300 p-6 rounded-lg">
                      <p className="text-3xl font-bold">{confusionMatrix[1][0]}</p>
                      <p className="text-xs text-muted-foreground">False Negatives</p>
                    </div>
                    <div className="bg-blue-100 border-2 border-blue-300 p-6 rounded-lg">
                      <p className="text-3xl font-bold">{confusionMatrix[1][1]}</p>
                      <p className="text-xs text-muted-foreground">True Positives</p>
                    </div>
                  </div>

                  <div className="mt-6 text-sm text-muted-foreground max-w-md">
                    <p>
                      <strong>True Negatives (TN):</strong> Normal points correctly identified
                    </p>
                    <p>
                      <strong>False Positives (FP):</strong> Normal points incorrectly flagged as anomalies
                    </p>
                    <p>
                      <strong>False Negatives (FN):</strong> Anomalies missed by the model
                    </p>
                    <p>
                      <strong>True Positives (TP):</strong> Anomalies correctly detected
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
}
