# React Visualization Components - Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Import Components
```tsx
// Common components
import {
  AlgorithmLayout,
  ParameterControl,
  CodeDisplay,
  ResultsPanel,
  LoadingSpinner,
  ErrorDisplay,
} from "@/components/common";

// Visualizations
import {
  LineChart,
  ScatterPlot,
  ConfusionMatrix,
  NetworkGraph,
} from "@/components/visualizations";

// Types
import type {
  ChartData,
  NetworkLayer,
  AlgorithmResults,
} from "@/types/visualizations";

// Utilities
import { calculateConfusionMatrix, formatNumber } from "@/lib/data-utils";
```

## Complete Example: Linear Regression

```tsx
import React, { useState } from "react";
import {
  AlgorithmLayout,
  ParameterControl,
  CodeDisplay,
  ResultsPanel,
  LoadingSpinner,
  ErrorDisplay,
} from "@/components/common";
import { LineChart, ScatterPlot } from "@/components/visualizations";
import type { ChartData } from "@/types/visualizations";

export function LinearRegressionDemo() {
  // State
  const [learningRate, setLearningRate] = useState(0.01);
  const [iterations, setIterations] = useState(100);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // Generate sample data
  const generateData = (): ChartData => {
    return Array.from({ length: 100 }, (_, i) => ({
      x: i,
      y: 2 * i + 10 + (Math.random() - 0.5) * 20,
      predicted: 2 * i + 10, // After training
    }));
  };

  const data = generateData();

  // Training history
  const trainingHistory = Array.from({ length: iterations }, (_, i) => ({
    iteration: i,
    loss: Math.exp(-i / 20) * 100,
  }));

  // Python code
  const code = `
import numpy as np
from sklearn.linear_model import LinearRegression

# Create model
model = LinearRegression()

# Training parameters
learning_rate = ${learningRate}
max_iterations = ${iterations}

# Fit the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
  `.trim();

  return (
    <AlgorithmLayout
      title="Linear Regression"
      description="Simple linear regression with gradient descent optimization"
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
              description="Step size for gradient descent"
            />

            <ParameterControl
              label="Iterations"
              value={iterations}
              onChange={(val) => setIterations(Number(val))}
              min={10}
              max={1000}
              step={10}
              type="slider"
              description="Number of training iterations"
            />
          </>
        ),

        visualization: (
          <div className="space-y-6">
            {isTraining ? (
              <LoadingSpinner message="Training model..." />
            ) : error ? (
              <ErrorDisplay
                error={error}
                title="Training Failed"
                retry={() => setError(null)}
              />
            ) : (
              <>
                <ScatterPlot
                  data={data}
                  xKey="x"
                  yKey="y"
                  title="Data Points and Regression Line"
                  xLabel="Feature (X)"
                  yLabel="Target (Y)"
                  height={400}
                />

                <LineChart
                  data={trainingHistory}
                  xKey="iteration"
                  yKey="loss"
                  title="Training Loss"
                  xLabel="Iteration"
                  yLabel="Loss"
                  height={300}
                />
              </>
            )}
          </div>
        ),

        code: <CodeDisplay code={code} language="python" />,

        results: (
          <ResultsPanel
            metrics={[
              {
                label: "R² Score",
                value: 0.94,
                format: (val) => val.toFixed(4),
                description: "Coefficient of determination",
              },
              {
                label: "MSE",
                value: 12.45,
                format: (val) => val.toFixed(2),
                description: "Mean squared error",
              },
              {
                label: "RMSE",
                value: 3.53,
                format: (val) => val.toFixed(2),
                description: "Root mean squared error",
              },
            ]}
          />
        ),
      }}
    />
  );
}
```

## Complete Example: K-Means Clustering

```tsx
import React, { useState } from "react";
import {
  AlgorithmLayout,
  ParameterControl,
  CodeDisplay,
  ResultsPanel,
} from "@/components/common";
import { ScatterPlot } from "@/components/visualizations";
import type { ClusterData } from "@/types/visualizations";

export function KMeansDemo() {
  const [k, setK] = useState(3);
  const [maxIter, setMaxIter] = useState(100);

  // Generate clustered data
  const generateClusterData = (): ClusterData => {
    const clusters: ClusterData = [];
    const centers = [
      { x: 2, y: 2 },
      { x: 8, y: 8 },
      { x: 2, y: 8 },
    ];

    centers.forEach((center, clusterIdx) => {
      for (let i = 0; i < 50; i++) {
        clusters.push({
          x: center.x + (Math.random() - 0.5) * 2,
          y: center.y + (Math.random() - 0.5) * 2,
          cluster: clusterIdx,
        });
      }
    });

    return clusters;
  };

  const data = generateClusterData();

  const code = `
from sklearn.cluster import KMeans
import numpy as np

# Create model
kmeans = KMeans(
    n_clusters=${k},
    max_iter=${maxIter},
    random_state=42
)

# Fit the model
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Calculate metrics
inertia = kmeans.inertia_
silhouette = silhouette_score(X, labels)
  `.trim();

  return (
    <AlgorithmLayout
      title="K-Means Clustering"
      description="Unsupervised clustering algorithm"
      sections={{
        parameters: (
          <>
            <ParameterControl
              label="Number of Clusters (k)"
              value={k}
              onChange={(val) => setK(Number(val))}
              min={2}
              max={10}
              step={1}
              type="slider"
            />

            <ParameterControl
              label="Max Iterations"
              value={maxIter}
              onChange={(val) => setMaxIter(Number(val))}
              min={10}
              max={500}
              step={10}
              type="slider"
            />
          </>
        ),

        visualization: (
          <ScatterPlot
            data={data}
            xKey="x"
            yKey="y"
            colorKey="cluster"
            title="K-Means Clustering Results"
            height={500}
            clusterNames={Array.from({ length: k }, (_, i) => `Cluster ${i + 1}`)}
          />
        ),

        code: <CodeDisplay code={code} language="python" />,

        results: (
          <ResultsPanel
            metrics={[
              {
                label: "Inertia",
                value: 234.56,
                format: (val) => val.toFixed(2),
                description: "Sum of squared distances to centroids",
              },
              {
                label: "Silhouette Score",
                value: 0.72,
                format: (val) => val.toFixed(3),
                description: "Clustering quality metric",
              },
            ]}
          />
        ),
      }}
    />
  );
}
```

## Complete Example: Neural Network

```tsx
import React, { useState } from "react";
import {
  AlgorithmLayout,
  ParameterControl,
  CodeDisplay,
  ResultsPanel,
} from "@/components/common";
import {
  NetworkGraph,
  LineChart,
  ConfusionMatrix,
} from "@/components/visualizations";
import type { NetworkLayer } from "@/types/visualizations";

export function NeuralNetworkDemo() {
  const [hiddenLayers, setHiddenLayers] = useState(2);
  const [neuronsPerLayer, setNeuronsPerLayer] = useState(8);
  const [activation, setActivation] = useState("relu");

  const layers: NetworkLayer[] = [
    { name: "Input", nodes: 4 },
    ...Array.from({ length: hiddenLayers }, (_, i) => ({
      name: `Hidden ${i + 1}`,
      nodes: neuronsPerLayer,
      activation,
    })),
    { name: "Output", nodes: 3 },
  ];

  const trainingData = Array.from({ length: 50 }, (_, i) => ({
    epoch: i,
    loss: Math.exp(-i / 10) * 2,
    accuracy: 1 - Math.exp(-i / 10) * 0.5,
  }));

  const confusionMatrix = [
    [45, 3, 2],
    [2, 46, 2],
    [1, 1, 48],
  ];

  const code = `
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# Create model
model = Sequential([
    Dense(4, activation='relu', input_shape=(4,)),
    ${Array.from({ length: hiddenLayers }, (_, i) => 
      `Dense(${neuronsPerLayer}, activation='${activation}')`
    ).join(',\n    ')},
    Dense(3, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)
  `.trim();

  return (
    <AlgorithmLayout
      title="Neural Network Classifier"
      description="Multi-layer perceptron for classification"
      sections={{
        parameters: (
          <>
            <ParameterControl
              label="Hidden Layers"
              value={hiddenLayers}
              onChange={(val) => setHiddenLayers(Number(val))}
              min={1}
              max={5}
              step={1}
              type="slider"
            />

            <ParameterControl
              label="Neurons per Layer"
              value={neuronsPerLayer}
              onChange={(val) => setNeuronsPerLayer(Number(val))}
              min={4}
              max={32}
              step={4}
              type="slider"
            />

            <ParameterControl
              label="Activation Function"
              value={activation}
              onChange={setActivation}
              type="select"
              options={[
                { value: "relu", label: "ReLU" },
                { value: "tanh", label: "Tanh" },
                { value: "sigmoid", label: "Sigmoid" },
              ]}
            />
          </>
        ),

        visualization: (
          <div className="space-y-6">
            <NetworkGraph
              layers={layers}
              title="Network Architecture"
              height={400}
            />

            <LineChart
              data={trainingData}
              xKey="epoch"
              yKey={["loss", "accuracy"]}
              title="Training Progress"
              height={300}
            />

            <ConfusionMatrix
              matrix={confusionMatrix}
              labels={["Class A", "Class B", "Class C"]}
            />
          </div>
        ),

        code: <CodeDisplay code={code} language="python" />,

        results: (
          <ResultsPanel
            metrics={[
              {
                label: "Accuracy",
                value: 0.946,
                format: (val) => `${(val * 100).toFixed(1)}%`,
              },
              {
                label: "Loss",
                value: 0.142,
                format: (val) => val.toFixed(3),
              },
              {
                label: "Parameters",
                value: 1000,
              },
            ]}
            predictions={[
              { label: "Class A", value: "45/50", confidence: 0.90 },
              { label: "Class B", value: "46/50", confidence: 0.92 },
              { label: "Class C", value: "48/50", confidence: 0.96 },
            ]}
          />
        ),
      }}
    />
  );
}
```

## Utility Functions Examples

```tsx
import {
  calculateConfusionMatrix,
  calculateAccuracy,
  calculateMetrics,
  movingAverage,
  trainTestSplit,
  formatNumber,
} from "@/lib/data-utils";

// Calculate confusion matrix
const predicted = [0, 1, 2, 0, 1, 2];
const actual = [0, 1, 1, 0, 1, 2];
const { matrix, labels } = calculateConfusionMatrix(predicted, actual);

// Calculate accuracy
const accuracy = calculateAccuracy(matrix);
console.log(`Accuracy: ${(accuracy * 100).toFixed(2)}%`);

// Calculate per-class metrics
const metrics = calculateMetrics(matrix, labels);
metrics.forEach(({ label, precision, recall, f1 }) => {
  console.log(`${label}: P=${precision.toFixed(2)}, R=${recall.toFixed(2)}, F1=${f1.toFixed(2)}`);
});

// Smooth noisy data
const noisyData = [1, 5, 2, 8, 3, 9, 4];
const smoothed = movingAverage(noisyData, 3);

// Split dataset
const { train, test } = trainTestSplit(data, 0.2, true);

// Format numbers
console.log(formatNumber(1234567)); // "1.23M"
console.log(formatNumber(5678, 1, "%")); // "5.7K%"
```

## Tips and Best Practices

1. **Always handle loading and error states**
2. **Use TypeScript for type safety**
3. **Keep visualizations responsive**
4. **Format numbers appropriately**
5. **Provide helpful descriptions**
6. **Test with different data sizes**
7. **Use meaningful color schemes**
8. **Add proper labels and titles**

## Common Patterns

### API Integration
```tsx
const [data, setData] = useState(null);
const [loading, setLoading] = useState(true);
const [error, setError] = useState(null);

useEffect(() => {
  fetchData()
    .then(setData)
    .catch(setError)
    .finally(() => setLoading(false));
}, []);

if (loading) return <LoadingSpinner />;
if (error) return <ErrorDisplay error={error} retry={fetchData} />;
return <YourVisualization data={data} />;
```

### Real-time Updates
```tsx
const [iteration, setIteration] = useState(0);
const [history, setHistory] = useState([]);

const train = () => {
  const interval = setInterval(() => {
    setIteration(i => i + 1);
    setHistory(h => [...h, { iteration, loss: Math.random() }]);
  }, 100);

  return () => clearInterval(interval);
};
```

## Resources

- Component Documentation: `/frontend/src/components/README.md`
- Example Code: `/frontend/src/components/examples/ComponentExamples.tsx`
- Type Definitions: `/frontend/src/types/visualizations.ts`
- Utility Functions: `/frontend/src/lib/data-utils.ts`
