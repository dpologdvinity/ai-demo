import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { FeatureImportance } from '@/components/visualizations/FeatureImportance';
import { ConfusionMatrix } from '@/components/visualizations/ConfusionMatrix';
import { RandomForestResults } from './index';
import { Loader2 } from 'lucide-react';

interface VisualizationProps {
  results: RandomForestResults | null;
  isTraining: boolean;
  error: string | null;
}

export const Visualization: React.FC<VisualizationProps> = ({
  results,
  isTraining,
  error,
}) => {
  if (isTraining) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center space-y-4">
          <Loader2 className="h-12 w-12 animate-spin mx-auto text-primary" />
          <p className="text-muted-foreground">Training Random Forest model...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center space-y-2">
          <p className="text-red-500 font-semibold">Error</p>
          <p className="text-muted-foreground">{error}</p>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center space-y-2">
          <p className="text-muted-foreground">
            Configure parameters and click "Train Model" to see results
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Accuracy</CardDescription>
            <CardTitle className="text-2xl">
              {(results.metrics.accuracy * 100).toFixed(2)}%
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Precision</CardDescription>
            <CardTitle className="text-2xl">
              {(results.metrics.precision * 100).toFixed(2)}%
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Recall</CardDescription>
            <CardTitle className="text-2xl">
              {(results.metrics.recall * 100).toFixed(2)}%
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>F1 Score</CardDescription>
            <CardTitle className="text-2xl">
              {(results.metrics.f1_score * 100).toFixed(2)}%
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Feature Importance */}
      <Card>
        <CardHeader>
          <CardTitle>Feature Importance</CardTitle>
          <CardDescription>
            Relative importance of each feature in making predictions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <FeatureImportance data={results.feature_importance} />
        </CardContent>
      </Card>

      {/* Confusion Matrix */}
      <Card>
        <CardHeader>
          <CardTitle>Confusion Matrix</CardTitle>
          <CardDescription>
            Classification results across all classes
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ConfusionMatrix
            matrix={results.confusion_matrix}
            labels={results.model_info.class_names}
          />
        </CardContent>
      </Card>

      {/* Model Info */}
      <Card>
        <CardHeader>
          <CardTitle>Model Information</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <dt className="font-semibold">Number of Trees</dt>
              <dd className="text-muted-foreground">{results.model_info.n_estimators}</dd>
            </div>
            <div>
              <dt className="font-semibold">Max Depth</dt>
              <dd className="text-muted-foreground">
                {results.model_info.max_depth === 'None' ? 'Unlimited' : results.model_info.max_depth}
              </dd>
            </div>
            <div>
              <dt className="font-semibold">Min Samples Split</dt>
              <dd className="text-muted-foreground">{results.model_info.min_samples_split}</dd>
            </div>
            <div>
              <dt className="font-semibold">Max Features</dt>
              <dd className="text-muted-foreground">{results.model_info.max_features}</dd>
            </div>
            <div>
              <dt className="font-semibold">Number of Features</dt>
              <dd className="text-muted-foreground">{results.model_info.n_features}</dd>
            </div>
            <div>
              <dt className="font-semibold">Number of Classes</dt>
              <dd className="text-muted-foreground">{results.model_info.n_classes}</dd>
            </div>
            <div className="col-span-2">
              <dt className="font-semibold">Execution Time</dt>
              <dd className="text-muted-foreground">
                {results.execution_time_ms.toFixed(2)} ms
              </dd>
            </div>
          </dl>
        </CardContent>
      </Card>
    </div>
  );
};
