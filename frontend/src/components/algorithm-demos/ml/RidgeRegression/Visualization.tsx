import { LineChart } from '@/components/visualizations/LineChart';
import { BarChart } from '@/components/visualizations/BarChart';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface VisualizationProps {
  result: {
    visualization_data: {
      predictions_vs_actual: Array<{
        index: number;
        predicted: number;
        actual: number;
      }>;
      coefficient_plot: Array<{
        feature: string;
        coefficient: number;
        coefficient_abs: number;
      }>;
      x_label: string;
      y_label: string;
      title: string;
    };
  };
}

export function Visualization({ result }: VisualizationProps) {
  const { predictions_vs_actual, coefficient_plot, x_label, y_label, title } =
    result.visualization_data;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Predictions vs Actual Values</CardTitle>
        </CardHeader>
        <CardContent>
          <LineChart
            data={predictions_vs_actual}
            xKey="index"
            yKey={['predicted', 'actual']}
            title={title}
            xLabel={x_label}
            yLabel={y_label}
            height={400}
            showGrid={true}
            showLegend={true}
            colors={['hsl(var(--primary))', 'hsl(142.1 76.2% 36.3%)']}
          />
          <div className="mt-4 text-sm text-muted-foreground text-center">
            <p>
              The chart shows predicted values (blue) vs actual values (green).
              Closer alignment indicates better model performance.
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Feature Coefficients</CardTitle>
        </CardHeader>
        <CardContent>
          <BarChart
            data={coefficient_plot}
            xKey="feature"
            yKey="coefficient"
            title="Ridge Regression Coefficients by Feature"
            xLabel="Features"
            yLabel="Coefficient Value"
            height={400}
            showGrid={true}
            colors={['hsl(var(--primary))']}
          />
          <div className="mt-4 text-sm text-muted-foreground text-center">
            <p>
              Feature coefficients show the impact of each feature on predictions.
              Ridge regularization shrinks coefficients to prevent overfitting.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
