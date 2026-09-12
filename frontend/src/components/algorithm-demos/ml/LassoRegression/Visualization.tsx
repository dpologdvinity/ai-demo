import { LineChart } from '@/components/visualizations/LineChart';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface VisualizationProps {
  result: {
    visualization_data: {
      chart_data: Array<{
        index: number;
        predicted: number;
        actual: number;
      }>;
      coefficient_data: Array<{
        feature: string;
        coefficient: number;
        selected: boolean;
      }>;
    };
  };
}

export function Visualization({ result }: VisualizationProps) {
  const { chart_data, coefficient_data } = result.visualization_data;

  return (
    <div className="space-y-6">
      {/* Predictions vs Actual */}
      <Card>
        <CardHeader>
          <CardTitle>Predictions vs Actual Values</CardTitle>
        </CardHeader>
        <CardContent>
          <LineChart
            data={chart_data}
            xKey="index"
            yKey={['predicted', 'actual']}
            title=""
            xLabel="Sample Index"
            yLabel="House Price (in $100k)"
            height={350}
            showGrid={true}
            showLegend={true}
            colors={['hsl(var(--primary))', 'hsl(142.1 76.2% 36.3%)']}
          />
          <div className="text-sm text-muted-foreground text-center mt-4">
            <p>
              The chart shows predicted values (blue) vs actual values (green).
              Closer alignment indicates better model performance.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Feature Coefficients showing sparsity */}
      <Card>
        <CardHeader>
          <CardTitle>Feature Coefficients (L1 Sparsity)</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart
              data={coefficient_data}
              margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
            >
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
              <XAxis
                dataKey="feature"
                angle={-45}
                textAnchor="end"
                height={100}
                tick={{ fontSize: 12 }}
              />
              <YAxis
                label={{ value: 'Coefficient Value', angle: -90, position: 'insideLeft' }}
                tick={{ fontSize: 12 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--background))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '6px',
                }}
                labelStyle={{ color: 'hsl(var(--foreground))' }}
              />
              <Legend />
              <Bar dataKey="coefficient" name="Coefficient" radius={[4, 4, 0, 0]}>
                {coefficient_data.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.selected ? 'hsl(var(--primary))' : 'hsl(var(--muted))'}
                    opacity={entry.selected ? 1 : 0.3}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="text-sm text-muted-foreground text-center mt-4 space-y-2">
            <p>
              <span className="inline-block w-3 h-3 rounded bg-primary mr-2"></span>
              <strong>Selected features</strong> have non-zero coefficients and contribute to predictions.
            </p>
            <p>
              <span className="inline-block w-3 h-3 rounded bg-muted mr-2 opacity-30"></span>
              <strong>Zero coefficients</strong> (faded) indicate features eliminated by L1 regularization.
            </p>
            <p className="pt-2">
              Lasso automatically performs feature selection by driving less important coefficients to exactly zero,
              creating a sparse model that's easier to interpret.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
