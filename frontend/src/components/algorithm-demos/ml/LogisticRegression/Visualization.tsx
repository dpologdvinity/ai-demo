import React from 'react';
import { ScatterPlot } from '@/components/visualizations/ScatterPlot';
import { ConfusionMatrix } from '@/components/visualizations/ConfusionMatrix';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/common/Card';

interface VisualizationProps {
  scatterData: {
    x: number[];
    y: number[];
    true_labels: number[];
    predicted_labels: number[];
    classes: any[];
    feature_names: string[];
  };
  confusionMatrix: {
    matrix: number[][];
    labels: string[];
    class_names: string[];
  };
  probabilityData: {
    samples: Array<{
      index: number;
      true_label: number;
      predicted_label: number;
      probabilities: Record<string, number>;
    }>;
  };
}

export function Visualization({
  scatterData,
  confusionMatrix,
  probabilityData,
}: VisualizationProps) {
  // Prepare scatter plot data
  const scatterPlotData = scatterData.x.map((x, i) => ({
    x: x,
    y: scatterData.y[i],
    label: scatterData.predicted_labels[i],
    correct: scatterData.true_labels[i] === scatterData.predicted_labels[i],
  }));

  // Group by predicted label for scatter plot
  const groupedData = scatterData.classes.map((cls) =>
    scatterPlotData.filter((point) => point.label === cls)
  );

  return (
    <div className="space-y-6">
      {/* Scatter Plot */}
      <ScatterPlot
        data={groupedData}
        xKey="x"
        yKey="y"
        title="Classification Results (First 2 Features)"
        xLabel={scatterData.feature_names[0] || 'Feature 1'}
        yLabel={scatterData.feature_names[1] || 'Feature 2'}
        clusterNames={scatterData.classes.map((cls) => `Class ${cls}`)}
        height={400}
      />

      {/* Confusion Matrix */}
      <ConfusionMatrix
        matrix={confusionMatrix.matrix}
        labels={confusionMatrix.class_names}
        title="Confusion Matrix"
      />

      {/* Top Predictions */}
      <Card>
        <CardHeader>
          <CardTitle>Top Confident Predictions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {probabilityData.samples.map((sample, idx) => (
              <div
                key={idx}
                className="border border-border rounded-lg p-3 space-y-2"
              >
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">
                    Sample #{sample.index}
                  </span>
                  <span
                    className={`text-xs px-2 py-1 rounded ${
                      sample.true_label === sample.predicted_label
                        ? 'bg-green-500/10 text-green-500'
                        : 'bg-red-500/10 text-red-500'
                    }`}
                  >
                    {sample.true_label === sample.predicted_label
                      ? 'Correct'
                      : 'Incorrect'}
                  </span>
                </div>

                <div className="text-xs text-muted-foreground">
                  True: Class {sample.true_label} | Predicted: Class{' '}
                  {sample.predicted_label}
                </div>

                <div className="space-y-1">
                  {Object.entries(sample.probabilities).map(([cls, prob]) => (
                    <div key={cls} className="flex items-center gap-2">
                      <span className="text-xs w-16">Class {cls}:</span>
                      <div className="flex-1 bg-muted rounded-full h-2">
                        <div
                          className="bg-primary h-2 rounded-full transition-all"
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                      <span className="text-xs font-medium w-12 text-right">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
