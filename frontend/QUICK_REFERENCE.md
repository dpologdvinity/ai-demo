# Quick Reference Card

## Installation
```bash
cd frontend
npm install
npm run dev
```

## Basic Template
```tsx
import {
  AlgorithmLayout,
  ParameterControl,
  CodeDisplay,
  ResultsPanel,
} from "@/components/common";
import { LineChart } from "@/components/visualizations";

export function MyAlgorithm() {
  const [param, setParam] = useState(0.5);
  
  return (
    <AlgorithmLayout
      title="My Algorithm"
      description="What it does"
      sections={{
        parameters: (
          <ParameterControl
            label="Parameter"
            value={param}
            onChange={setParam}
            min={0}
            max={1}
            step={0.1}
            type="slider"
          />
        ),
        visualization: <LineChart data={data} xKey="x" yKey="y" />,
        code: <CodeDisplay code={code} language="python" />,
        results: <ResultsPanel metrics={metrics} />
      }}
    />
  );
}
```

## Component Cheat Sheet

### ParameterControl
```tsx
<ParameterControl
  label="Name"
  value={value}
  onChange={setValue}
  type="slider" // or "number" or "select"
  min={0} max={100} step={1}
  options={[{value: "a", label: "A"}]} // for select
/>
```

### LineChart
```tsx
<LineChart
  data={[{x: 1, y: 2}, ...]}
  xKey="x"
  yKey="y" // or ["y1", "y2"] for multiple
  title="Chart Title"
  height={400}
/>
```

### ScatterPlot
```tsx
<ScatterPlot
  data={[{x: 1, y: 2, cluster: 0}, ...]}
  xKey="x"
  yKey="y"
  colorKey="cluster" // optional, for clustering
  clusterNames={["A", "B"]}
/>
```

### ConfusionMatrix
```tsx
<ConfusionMatrix
  matrix={[[45, 5], [3, 47]]}
  labels={["Class A", "Class B"]}
/>
```

### NetworkGraph
```tsx
<NetworkGraph
  layers={[
    {name: "Input", nodes: 4},
    {name: "Hidden", nodes: 8},
    {name: "Output", nodes: 3}
  ]}
  showWeights={false}
/>
```

### CodeDisplay
```tsx
<CodeDisplay
  code={pythonCode}
  language="python"
  showLineNumbers={true}
/>
```

### ResultsPanel
```tsx
<ResultsPanel
  metrics={[
    {label: "Accuracy", value: 0.94, format: v => `${v*100}%`}
  ]}
  predictions={[
    {label: "Class A", value: "45/50", confidence: 0.9}
  ]}
/>
```

### LoadingSpinner
```tsx
<LoadingSpinner message="Loading..." size="md" />
```

### ErrorDisplay
```tsx
<ErrorDisplay
  error={error}
  retry={() => handleRetry()}
/>
```

## Utility Functions
```tsx
import {
  calculateConfusionMatrix,
  calculateAccuracy,
  normalize,
  movingAverage,
  trainTestSplit,
  formatNumber,
} from "@/lib/data-utils";

// Confusion matrix
const {matrix, labels} = calculateConfusionMatrix(pred, actual);

// Accuracy
const acc = calculateAccuracy(matrix);

// Normalize
const normalized = normalize([1, 2, 3, 4, 5]);

// Smooth data
const smooth = movingAverage(data, windowSize);

// Split data
const {train, test} = trainTestSplit(data, 0.2);

// Format numbers
formatNumber(1234567) // "1.23M"
```

## Common Patterns

### API Call with Loading/Error
```tsx
const [data, setData] = useState(null);
const [loading, setLoading] = useState(true);
const [error, setError] = useState(null);

useEffect(() => {
  fetch('/api/data')
    .then(r => r.json())
    .then(setData)
    .catch(setError)
    .finally(() => setLoading(false));
}, []);

if (loading) return <LoadingSpinner />;
if (error) return <ErrorDisplay error={error} />;
return <YourComponent data={data} />;
```

### Real-time Training
```tsx
const [history, setHistory] = useState([]);

const train = async () => {
  for (let i = 0; i < epochs; i++) {
    const loss = await trainStep(i);
    setHistory(h => [...h, {epoch: i, loss}]);
  }
};
```

## Types
```tsx
import type {
  ChartData,
  ClusterData,
  NetworkLayer,
  TrainingMetrics,
  AlgorithmResults,
} from "@/types/visualizations";

const data: ChartData = [{x: 1, y: 2}];
const layers: NetworkLayer[] = [{name: "Input", nodes: 4}];
```

## Styling
```tsx
// Add custom classes
<LineChart className="my-4 border rounded" ... />

// Use cn() utility for conditional classes
import { cn } from "@/lib/utils";

<div className={cn(
  "base-class",
  isActive && "active-class",
  className
)}>
```

## Documentation
- Components: `/frontend/src/components/README.md`
- Usage: `/USAGE_GUIDE.md`
- Setup: `/SETUP_CHECKLIST.md`
- Examples: `/frontend/src/components/examples/`
