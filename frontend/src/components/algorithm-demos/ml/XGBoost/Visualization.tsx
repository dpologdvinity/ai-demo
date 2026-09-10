import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart } from '@/components/visualizations/LineChart';
import { ConfusionMatrix } from '@/components/visualizations/ConfusionMatrix';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { ErrorMessage } from '@/components/common/ErrorMessage';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface XGBoostResult {
  success: boolean;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
  predictions: number[];
  visualization_data: {
    confusion_matrix: number[][];
    feature_importance: {
      features: string[];
      importance: number[];
    };
    learning_curves: {
      n_estimators: number[];
      train_accuracy: number[];
      test_accuracy: number[];
    };
    class_probabilities: number[][];
    target_names: string[];
  };
  execution_time_ms: number;
  parameters_used: Record<string, any>;
}

interface VisualizationProps {
  result: XGBoostResult | null;
  isTraining: boolean;
  error: string | null;
}

export function Visualization({ result, isTraining, error }: VisualizationProps) {
  if (isTraining) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return <ErrorMessage message={error} />;
  }

  if (!result) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        <p>Configure parameters and click "Train Model" to see results</p>
      </div>
    );
  }

  // Prepare feature importance data
  const featureImportanceData =
    result.visualization_data.feature_importance.features.map((feature, index) => ({
      feature: feature.length > 20 ? feature.substring(0, 20) + '...' : feature,
      importance: result.visualization_data.feature_importance.importance[index],
    }));

  // Prepare learning curves data
  const learningCurvesData =
    result.visualization_data.learning_curves.n_estimators.map((n, index) => ({
      n_estimators: n,
      train_accuracy: result.visualization_data.learning_curves.train_accuracy[index],
      test_accuracy: result.visualization_data.learning_curves.test_accuracy[index],
    }));

  return (
    <div className="space-y-6">
      {/* Metrics */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Accuracy</p>
              <p className="text-2xl font-bold">
                {(result.metrics.accuracy * 100).toFixed(2)}%
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Precision</p>
              <p className="text-2xl font-bold">
                {(result.metrics.precision * 100).toFixed(2)}%
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Recall</p>
              <p className="text-2xl font-bold">
                {(result.metrics.recall * 100).toFixed(2)}%
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">F1 Score</p>
              <p className="text-2xl font-bold">
                {(result.metrics.f1_score * 100).toFixed(2)}%
              </p>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t">
            <p className="text-sm text-muted-foreground">
              Training time: {result.execution_time_ms.toFixed(2)}ms
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Feature Importance */}
      <Card>
        <CardHeader>
          <CardTitle>Feature Importance</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={featureImportanceData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" opacity={0.3} />
              <XAxis type="number" className="text-muted-foreground" />
              <YAxis
                dataKey="feature"
                type="category"
                className="text-muted-foreground text-xs"
                width={90}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--popover))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: 'var(--radius)',
                }}
              />
              <Bar dataKey="importance" fill="hsl(var(--primary))" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Learning Curves */}
      <LineChart
        data={learningCurvesData}
        xKey="n_estimators"
        yKey={['train_accuracy', 'test_accuracy']}
        title="Learning Curves"
        xLabel="Number of Estimators"
        yLabel="Accuracy"
        height={300}
      />

      {/* Confusion Matrix */}
      <ConfusionMatrix
        matrix={result.visualization_data.confusion_matrix}
        labels={result.visualization_data.target_names}
        title="Confusion Matrix"
      />
    </div>
  );
}
