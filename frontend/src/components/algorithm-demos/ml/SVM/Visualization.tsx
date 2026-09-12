import { ScatterPlot } from '@/components/visualizations/ScatterPlot';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DataPoint {
  x: number;
  y: number;
  label: number;
  predicted: number;
  is_support_vector: boolean;
  type: string;
}

interface SVMMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: number[][];
}

interface VisualizationData {
  training_data: DataPoint[];
  test_data: DataPoint[];
  support_vectors: Array<{
    x: number;
    y: number;
    is_support_vector: boolean;
  }>;
  decision_boundary: any;
  feature_names: string[];
  target_names: string[];
}

interface VisualizationProps {
  result: {
    metrics: SVMMetrics;
    model_info: any;
    visualization_data: VisualizationData;
    execution_time_ms: number;
  };
}

const classColors = ['#3b82f6', '#10b981', '#f59e0b']; // blue, green, amber

export function Visualization({ result }: VisualizationProps) {
  const { visualization_data } = result;

  // Combine training and test data for visualization
  const allData = [
    ...visualization_data.training_data,
    ...visualization_data.test_data,
  ];

  // Group data by class for scatter plot
  const dataByClass: DataPoint[][] = [];
  const classes = [...new Set(allData.map((d) => d.label))].sort();

  classes.forEach((classLabel) => {
    const classData = allData.filter((d) => d.label === classLabel);
    dataByClass.push(classData);
  });

  // Create separate dataset for support vectors (highlighted)
  const supportVectorData = visualization_data.training_data
    .filter((d) => d.is_support_vector)
    .map((d) => ({
      ...d,
      // Mark support vectors with a special identifier
      isSV: true,
    }));

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Decision Boundary & Support Vectors</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <ScatterPlot
              data={dataByClass}
              xKey="x"
              yKey="y"
              xLabel={visualization_data.feature_names[0] || 'Feature 1'}
              yLabel={visualization_data.feature_names[1] || 'Feature 2'}
              height={500}
              colors={classColors}
              clusterNames={visualization_data.target_names}
              showGrid={true}
              showLegend={true}
            />

            {/* Support vectors overlay - shown as larger points */}
            {supportVectorData.length > 0 && (
              <div className="mt-4 p-4 border-t">
                <p className="text-sm font-medium mb-2">
                  Support Vectors (outlined points)
                </p>
                <p className="text-xs text-muted-foreground">
                  {supportVectorData.length} data points are support vectors (closest to decision boundary).
                  These are the critical points that define the hyperplane.
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Training vs Test Data</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm font-medium mb-2">Training Data</p>
              <div className="space-y-1">
                {classes.map((classLabel, idx) => {
                  const count = visualization_data.training_data.filter(
                    (d) => d.label === classLabel
                  ).length;
                  return (
                    <div key={classLabel} className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: classColors[idx] }}
                        />
                        <span>{visualization_data.target_names[idx]}</span>
                      </div>
                      <span className="text-muted-foreground">{count}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div>
              <p className="text-sm font-medium mb-2">Test Data</p>
              <div className="space-y-1">
                {classes.map((classLabel, idx) => {
                  const count = visualization_data.test_data.filter(
                    (d) => d.label === classLabel
                  ).length;
                  return (
                    <div key={classLabel} className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: classColors[idx] }}
                        />
                        <span>{visualization_data.target_names[idx]}</span>
                      </div>
                      <span className="text-muted-foreground">{count}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Classification Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <p className="text-sm text-muted-foreground mb-2">Accuracy by Class</p>
              {classes.map((classLabel, idx) => {
                const classTestData = visualization_data.test_data.filter(
                  (d) => d.label === classLabel
                );
                const correct = classTestData.filter(
                  (d) => d.label === d.predicted
                ).length;
                const accuracy = (correct / classTestData.length) * 100;

                return (
                  <div key={classLabel} className="mb-2">
                    <div className="flex justify-between text-sm mb-1">
                      <span>{visualization_data.target_names[idx]}</span>
                      <span className="font-medium">{accuracy.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                      <div
                        className="h-2 rounded-full transition-all"
                        style={{
                          width: `${accuracy}%`,
                          backgroundColor: classColors[idx],
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="pt-4 border-t">
              <p className="text-sm font-medium mb-2">Misclassifications</p>
              {(() => {
                const misclassified = visualization_data.test_data.filter(
                  (d) => d.label !== d.predicted
                );

                if (misclassified.length === 0) {
                  return (
                    <p className="text-sm text-green-600 dark:text-green-400">
                      Perfect classification! No misclassifications.
                    </p>
                  );
                }

                return (
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">
                      {misclassified.length} test samples were misclassified
                    </p>
                    <div className="max-h-32 overflow-y-auto text-xs space-y-1">
                      {misclassified.slice(0, 10).map((d, idx) => (
                        <div key={idx} className="text-muted-foreground">
                          Predicted {visualization_data.target_names[d.predicted]} but was{' '}
                          {visualization_data.target_names[d.label]}
                        </div>
                      ))}
                      {misclassified.length > 10 && (
                        <div className="text-muted-foreground italic">
                          ... and {misclassified.length - 10} more
                        </div>
                      )}
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
