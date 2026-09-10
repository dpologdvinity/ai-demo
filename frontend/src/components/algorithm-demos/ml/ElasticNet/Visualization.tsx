import { LineChart } from '@/components/visualizations/LineChart';
import { LineChart as RechartsLineChart, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, Line } from 'recharts';
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
        magnitude: number;
        selected: boolean;
      }>;
      regularization_path: Array<{
        l1_ratio: number;
        type: string;
        r2_score: number;
        n_nonzero: number;
        sparsity: number;
        l2_norm: number;
      }>;
    };
  };
}

export function Visualization({ result }: VisualizationProps) {
  const { chart_data, coefficient_data, regularization_path } = result.visualization_data;

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

      {/* Feature Coefficients */}
      <Card>
        <CardHeader>
          <CardTitle>Feature Coefficients (Elastic Net Regularization)</CardTitle>
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
              <strong>Zero coefficients</strong> (faded) indicate features eliminated by L1 component.
            </p>
            <p className="pt-2">
              Elastic Net combines L1 regularization (feature selection) with L2 regularization (coefficient shrinkage),
              providing a balance between sparsity and stability.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Regularization Path: Ridge ↔ Elastic Net ↔ Lasso */}
      <Card>
        <CardHeader>
          <CardTitle>Regularization Path: Ridge ↔ Elastic Net ↔ Lasso</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* R² Score across l1_ratio */}
            <div>
              <p className="text-sm font-medium mb-2">Performance (R² Score)</p>
              <ResponsiveContainer width="100%" height={200}>
                <RechartsLineChart data={regularization_path}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis
                    dataKey="l1_ratio"
                    label={{ value: 'L1 Ratio', position: 'insideBottom', offset: -5 }}
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => value.toFixed(2)}
                  />
                  <YAxis
                    label={{ value: 'R² Score', angle: -90, position: 'insideLeft' }}
                    tick={{ fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--background))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '6px',
                    }}
                    labelStyle={{ color: 'hsl(var(--foreground))' }}
                    formatter={(value: any) => [value.toFixed(4), 'R² Score']}
                    labelFormatter={(value) => `L1 Ratio: ${parseFloat(value).toFixed(2)}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="r2_score"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                  />
                </RechartsLineChart>
              </ResponsiveContainer>
            </div>

            {/* Sparsity across l1_ratio */}
            <div>
              <p className="text-sm font-medium mb-2">Sparsity (Feature Selection)</p>
              <ResponsiveContainer width="100%" height={200}>
                <RechartsLineChart data={regularization_path}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis
                    dataKey="l1_ratio"
                    label={{ value: 'L1 Ratio', position: 'insideBottom', offset: -5 }}
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => value.toFixed(2)}
                  />
                  <YAxis
                    label={{ value: 'Sparsity (%)', angle: -90, position: 'insideLeft' }}
                    tick={{ fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--background))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '6px',
                    }}
                    labelStyle={{ color: 'hsl(var(--foreground))' }}
                    formatter={(value: any) => [value.toFixed(1) + '%', 'Sparsity']}
                    labelFormatter={(value) => `L1 Ratio: ${parseFloat(value).toFixed(2)}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="sparsity"
                    stroke="hsl(142.1 76.2% 36.3%)"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                  />
                </RechartsLineChart>
              </ResponsiveContainer>
            </div>

            <div className="text-sm text-muted-foreground space-y-2 pt-4 border-t">
              <p className="flex items-center">
                <span className="font-semibold w-32">L1 Ratio = 0.0:</span>
                <span>Pure Ridge (L2) - No sparsity, all features retained with shrinkage</span>
              </p>
              <p className="flex items-center">
                <span className="font-semibold w-32">L1 Ratio = 0.5:</span>
                <span>Elastic Net - Balanced mix of L1 and L2 regularization</span>
              </p>
              <p className="flex items-center">
                <span className="font-semibold w-32">L1 Ratio = 1.0:</span>
                <span>Pure Lasso (L1) - Maximum sparsity, aggressive feature selection</span>
              </p>
              <p className="pt-2 italic">
                As L1 ratio increases, the model becomes more sparse (selects fewer features) but may lose some predictive power.
                Elastic Net finds the optimal balance for your specific problem.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
