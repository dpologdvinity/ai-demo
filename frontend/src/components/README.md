# Reusable React Components for AI Algorithm Visualizations

This directory contains reusable React components for building AI algorithm demonstrations with consistent styling, responsive layouts, and theme support.

## Directory Structure

```
components/
├── common/              # Common UI components
│   ├── ParameterControl.tsx
│   ├── CodeDisplay.tsx
│   ├── AlgorithmLayout.tsx
│   ├── LoadingSpinner.tsx
│   ├── ErrorDisplay.tsx
│   └── ResultsPanel.tsx
├── visualizations/      # Data visualization components
│   ├── LineChart.tsx
│   ├── ScatterPlot.tsx
│   ├── ConfusionMatrix.tsx
│   └── NetworkGraph.tsx
├── ui/                  # Base UI components (Shadcn/UI)
│   ├── button.tsx
│   ├── card.tsx
│   ├── input.tsx
│   ├── slider.tsx
│   └── select.tsx
└── examples/            # Example usage
    └── ComponentExamples.tsx
```

## Components Overview

### Common Components

#### 1. ParameterControl
Reusable component for algorithm parameters with multiple input types.

```tsx
import { ParameterControl } from "@/components/common";

<ParameterControl
  label="Learning Rate"
  value={learningRate}
  onChange={setLearningRate}
  min={0.001}
  max={0.1}
  step={0.001}
  type="slider"  // or "number" or "select"
  description="Controls how quickly the model adapts"
/>
```

**Props:**
- `label`: Parameter label
- `value`: Current value
- `onChange`: Callback when value changes
- `min`, `max`, `step`: Numeric constraints
- `type`: "slider", "number", or "select"
- `options`: Array of options for select type
- `description`: Optional description text

#### 2. CodeDisplay
Syntax highlighted code display with copy functionality.

```tsx
import { CodeDisplay } from "@/components/common";

<CodeDisplay
  code={pythonCode}
  language="python"
  title="Implementation"
  showLineNumbers={true}
/>
```

**Features:**
- Automatic theme detection (light/dark)
- Copy to clipboard button
- Line numbers
- Multiple language support

#### 3. AlgorithmLayout
Standard layout for algorithm demos with consistent sections.

```tsx
import { AlgorithmLayout } from "@/components/common";

<AlgorithmLayout
  title="Neural Network"
  description="Train a classifier"
  sections={{
    parameters: <ParameterControls />,
    visualization: <Charts />,
    code: <CodeDisplay />,
    results: <ResultsPanel />
  }}
/>
```

**Sections:**
- `parameters`: Left sidebar with controls
- `code`: Code implementation
- `visualization`: Main visualization area
- `results`: Results and metrics

#### 4. LoadingSpinner
Loading state component with optional message.

```tsx
import { LoadingSpinner } from "@/components/common";

<LoadingSpinner message="Training model..." size="md" />
```

#### 5. ErrorDisplay
Error state component with retry functionality.

```tsx
import { ErrorDisplay } from "@/components/common";

<ErrorDisplay
  error={error}
  title="Training Failed"
  retry={() => handleRetry()}
/>
```

#### 6. ResultsPanel
Display algorithm results with metrics and predictions.

```tsx
import { ResultsPanel } from "@/components/common";

<ResultsPanel
  metrics={[
    {
      label: "Accuracy",
      value: 0.94,
      format: (val) => `${(val * 100).toFixed(2)}%`,
      description: "Overall accuracy"
    }
  ]}
  predictions={[
    { label: "Class A", value: "15/18", confidence: 0.83 }
  ]}
/>
```

### Visualization Components

#### 1. LineChart
Wrapper around Recharts LineChart with theme support.

```tsx
import { LineChart } from "@/components/visualizations";

<LineChart
  data={trainingData}
  xKey="epoch"
  yKey={["loss", "accuracy"]}  // Multiple lines
  title="Training Progress"
  xLabel="Epoch"
  yLabel="Value"
  height={400}
/>
```

#### 2. ScatterPlot
Scatter plot with clustering support.

```tsx
import { ScatterPlot } from "@/components/visualizations";

<ScatterPlot
  data={points}
  xKey="x"
  yKey="y"
  colorKey="cluster"  // For clustering visualization
  title="Data Distribution"
  clusterNames={["Class A", "Class B"]}
/>
```

#### 3. ConfusionMatrix
Heatmap for classification results.

```tsx
import { ConfusionMatrix } from "@/components/visualizations";

<ConfusionMatrix
  matrix={[
    [45, 3, 2],
    [1, 48, 1],
    [2, 2, 46]
  ]}
  labels={["Class A", "Class B", "Class C"]}
/>
```

#### 4. NetworkGraph
3D neural network visualization using React Three Fiber.

```tsx
import { NetworkGraph } from "@/components/visualizations";

<NetworkGraph
  layers={[
    { name: "Input", nodes: 4 },
    { name: "Hidden", nodes: 8 },
    { name: "Output", nodes: 3 }
  ]}
  weights={weightsArray}  // Optional
  showWeights={true}
  height={500}
/>
```

**Features:**
- Interactive 3D visualization
- Mouse controls (rotate, zoom, pan)
- Optional weight visualization
- Layer labels

## Features

All components include:
- **TypeScript**: Full type safety with proper prop types
- **Tailwind CSS**: Utility-first styling
- **Responsive**: Mobile-friendly layouts
- **Theme Support**: Automatic light/dark mode
- **Accessibility**: Semantic HTML and ARIA labels

## Installation

Required dependencies are already in `package.json`:

```bash
npm install
```

Key dependencies:
- `recharts`: Charts and graphs
- `react-syntax-highlighter`: Code highlighting
- `three`, `@react-three/fiber`, `@react-three/drei`: 3D visualization
- `@radix-ui/*`: Accessible UI primitives
- `lucide-react`: Icons

## Usage Example

See `/components/examples/ComponentExamples.tsx` for a complete working example that demonstrates all components together.

## Styling

Components use Tailwind CSS with CSS variables for theming. Colors are defined in `tailwind.config.js` and can be customized through CSS variables:

```css
:root {
  --primary: ...;
  --secondary: ...;
  --background: ...;
  /* etc */
}
```

## Best Practices

1. **Import from index files**: Use barrel exports for cleaner imports
   ```tsx
   import { LineChart, ScatterPlot } from "@/components/visualizations";
   ```

2. **Use AlgorithmLayout**: For consistent page structure
3. **Handle loading/error states**: Use LoadingSpinner and ErrorDisplay
4. **Responsive design**: Components adapt to screen size
5. **Theme awareness**: Components automatically adapt to light/dark mode

## Customization

All components accept a `className` prop for custom styling:

```tsx
<LineChart
  data={data}
  xKey="x"
  yKey="y"
  className="my-custom-class"
/>
```

Use the `cn()` utility from `@/lib/utils` to merge Tailwind classes safely.
