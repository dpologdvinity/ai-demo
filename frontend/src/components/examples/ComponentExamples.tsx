import * as React from "react";
import {
  AlgorithmLayout,
  ParameterControl,
  CodeDisplay,
  ResultsPanel,
  LoadingSpinner,
  ErrorDisplay,
} from "@/components/common";
import {
  LineChart,
  ScatterPlot,
  ConfusionMatrix,
  NetworkGraph,
} from "@/components/visualizations";

/**
 * Example demonstrating how to use all the reusable components together
 */
export function ComponentExamples() {
  const [learningRate, setLearningRate] = React.useState(0.01);
  const [epochs, setEpochs] = React.useState(100);
  const [optimizer, setOptimizer] = React.useState("adam");
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);

  // Example data for visualizations
  const lineChartData = Array.from({ length: 50 }, (_, i) => ({
    epoch: i,
    loss: Math.exp(-i / 10) + Math.random() * 0.1,
    accuracy: 1 - Math.exp(-i / 10) + Math.random() * 0.05,
  }));

  const scatterData = Array.from({ length: 100 }, (_, i) => ({
    x: Math.random() * 10,
    y: Math.random() * 10,
    cluster: Math.floor(i / 25),
  }));

  const confusionMatrixData = [
    [45, 3, 2],
    [1, 48, 1],
    [2, 2, 46],
  ];

  const networkLayers = [
    { name: "Input", nodes: 4 },
    { name: "Hidden 1", nodes: 8 },
    { name: "Hidden 2", nodes: 6 },
    { name: "Output", nodes: 3 },
  ];

  const exampleCode = `
import numpy as np
from sklearn.neural_network import MLPClassifier

# Create and train the model
model = MLPClassifier(
    hidden_layer_sizes=(8, 6),
    learning_rate_init=${learningRate},
    max_iter=${epochs},
    solver='${optimizer}'
)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
  `.trim();

  return (
    <AlgorithmLayout
      title="Neural Network Classification"
      description="Train a multi-layer perceptron classifier with customizable parameters"
      sections={{
        parameters: (
          <>
            <ParameterControl
              label="Learning Rate"
              value={learningRate}
              onChange={(val) => setLearningRate(Number(val))}
              min={0.001}
              max={0.1}
              step={0.001}
              type="slider"
              description="Controls how quickly the model adapts to the problem"
            />

            <ParameterControl
              label="Epochs"
              value={epochs}
              onChange={(val) => setEpochs(Number(val))}
              min={10}
              max={1000}
              step={10}
              type="slider"
              description="Number of training iterations"
            />

            <ParameterControl
              label="Optimizer"
              value={optimizer}
              onChange={setOptimizer}
              type="select"
              options={[
                { value: "adam", label: "Adam" },
                { value: "sgd", label: "SGD" },
                { value: "rmsprop", label: "RMSprop" },
              ]}
              description="Optimization algorithm to use"
            />
          </>
        ),

        code: (
          <CodeDisplay
            code={exampleCode}
            language="python"
            showLineNumbers={true}
          />
        ),

        visualization: (
          <div className="space-y-6">
            {isLoading ? (
              <LoadingSpinner message="Training model..." />
            ) : error ? (
              <ErrorDisplay
                error={error}
                title="Training Failed"
                retry={() => {
                  setError(null);
                  setIsLoading(false);
                }}
              />
            ) : (
              <>
                <LineChart
                  data={lineChartData}
                  xKey="epoch"
                  yKey={["loss", "accuracy"]}
                  title="Training Progress"
                  xLabel="Epoch"
                  yLabel="Value"
                  height={300}
                />

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <ScatterPlot
                    data={scatterData}
                    xKey="x"
                    yKey="y"
                    colorKey="cluster"
                    title="Data Distribution"
                    height={300}
                    clusterNames={["Class A", "Class B", "Class C", "Class D"]}
                  />

                  <ConfusionMatrix
                    matrix={confusionMatrixData}
                    labels={["Class A", "Class B", "Class C"]}
                    title="Confusion Matrix"
                  />
                </div>

                <NetworkGraph
                  layers={networkLayers}
                  title="Network Architecture"
                  height={400}
                  showWeights={false}
                />
              </>
            )}
          </div>
        ),

        results: (
          <ResultsPanel
            metrics={[
              {
                label: "Accuracy",
                value: 0.94,
                format: (val) => `${(val * 100).toFixed(2)}%`,
                description: "Overall classification accuracy",
              },
              {
                label: "Precision",
                value: 0.92,
                format: (val) => `${(val * 100).toFixed(2)}%`,
              },
              {
                label: "Recall",
                value: 0.93,
                format: (val) => `${(val * 100).toFixed(2)}%`,
              },
              {
                label: "F1 Score",
                value: 0.925,
                format: (val) => val.toFixed(3),
              },
              {
                label: "Training Time",
                value: "2.3s",
              },
              {
                label: "Parameters",
                value: 156,
              },
            ]}
            predictions={[
              { label: "Class A", value: "15/18", confidence: 0.83 },
              { label: "Class B", value: "16/17", confidence: 0.94 },
              { label: "Class C", value: "14/15", confidence: 0.93 },
            ]}
          />
        ),
      }}
    />
  );
}
