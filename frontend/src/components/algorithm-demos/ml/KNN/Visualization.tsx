import { ScatterPlot } from '@/components/visualizations/ScatterPlot';
import { ConfusionMatrix } from '@/components/visualizations/ConfusionMatrix';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface VisualizationProps {
  result: {
    visualization_data: {
      scatter_data: Array<{
        x: number;
        y: number;
        class: number;
        predicted: number;
        label: string;
      }>;
      confusion_matrix: number[][];
      class_labels: string[];
      decision_boundary?: Array<{
        x: number;
        y: number;
        class: number;
      }>;
    };
  };
}

export function Visualization({ result }: VisualizationProps) {
  const { scatter_data, confusion_matrix, class_labels, decision_boundary } =
    result.visualization_data;

  // Group scatter data by class for multi-colored scatter plot
  const scatterByClass: Array<Array<{ x: number; y: number }>> = [];
  const maxClass = Math.max(...scatter_data.map((d) => d.class));

  for (let i = 0; i <= maxClass; i++) {
    scatterByClass.push(
      scatter_data
        .filter((d) => d.class === i)
        .map((d) => ({ x: d.x, y: d.y }))
    );
  }

  // Prepare decision boundary data if available
  let boundaryByClass: Array<Array<{ x: number; y: number }>> = [];
  if (decision_boundary && decision_boundary.length > 0) {
    const maxBoundaryClass = Math.max(...decision_boundary.map((d) => d.class));
    for (let i = 0; i <= maxBoundaryClass; i++) {
      boundaryByClass.push(
        decision_boundary
          .filter((d) => d.class === i)
          .map((d) => ({ x: d.x, y: d.y }))
      );
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>2D Feature Space (PCA Projection)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative">
            {/* Decision boundary background (if available) */}
            {boundaryByClass.length > 0 && (
              <div className="absolute inset-0 opacity-20 pointer-events-none">
                <ScatterPlot
                  data={boundaryByClass}
                  xKey="x"
                  yKey="y"
                  xLabel="Principal Component 1"
                  yLabel="Principal Component 2"
                  height={400}
                  showGrid={false}
                  showLegend={false}
                  clusterNames={class_labels}
                  colors={['#8884d8', '#82ca9d', '#ffc658']}
                />
              </div>
            )}
            {/* Actual data points */}
            <ScatterPlot
              data={scatterByClass}
              xKey="x"
              yKey="y"
              xLabel="Principal Component 1"
              yLabel="Principal Component 2"
              height={400}
              showGrid={true}
              showLegend={true}
              clusterNames={class_labels}
              colors={['#8884d8', '#82ca9d', '#ffc658']}
            />
          </div>
          <p className="text-sm text-muted-foreground mt-4">
            The Iris dataset has 4 features. This visualization shows the data projected
            onto 2 dimensions using Principal Component Analysis (PCA) for visualization
            purposes. Colors represent the actual classes.
          </p>
        </CardContent>
      </Card>

      <ConfusionMatrix
        matrix={confusion_matrix}
        labels={class_labels}
        title="Confusion Matrix"
      />

      <Card>
        <CardHeader>
          <CardTitle>Classification Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-sm">
            <p>
              <span className="font-medium">Total Samples:</span>{' '}
              {scatter_data.length}
            </p>
            <p>
              <span className="font-medium">Number of Classes:</span>{' '}
              {class_labels.length}
            </p>
            <p>
              <span className="font-medium">Classes:</span>{' '}
              {class_labels.join(', ')}
            </p>
            <p className="mt-4 text-muted-foreground">
              The confusion matrix shows the count of correct and incorrect predictions
              for each class. Diagonal cells represent correct predictions.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
