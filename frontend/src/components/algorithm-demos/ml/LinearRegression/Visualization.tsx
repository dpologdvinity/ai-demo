import { LineChart } from '@/components/visualizations/LineChart';

interface VisualizationProps {
  result: {
    visualization_data: {
      chart_data: Array<{
        index: number;
        predicted: number;
        actual: number;
      }>;
      x_label: string;
      y_label: string;
      title: string;
    };
  };
}

export function Visualization({ result }: VisualizationProps) {
  const { chart_data, x_label, y_label, title } = result.visualization_data;

  return (
    <div className="space-y-4">
      <LineChart
        data={chart_data}
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

      <div className="text-sm text-muted-foreground text-center">
        <p>
          The chart shows predicted values (blue) vs actual values (green).
          Closer alignment indicates better model performance.
        </p>
      </div>
    </div>
  );
}
