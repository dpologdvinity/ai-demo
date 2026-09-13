import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { AlgorithmLayout } from '@/components/common/AlgorithmLayout';
import { apiService } from '@/services/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { StepPlayer, useStepPlayback } from '@/components/common/StepPlayer';
import { MathPanel } from '@/components/common/MathPanel';
import Controls from './Controls';
import Visualization from './Visualization';
import Documentation from './Documentation';
import { Loader2 } from 'lucide-react';

interface AStarParameters {
  grid_size: number;
  obstacle_density: number;
  heuristic: 'manhattan' | 'euclidean' | 'chebyshev';
  random_state: number;
}

interface StepTrace {
  step: number;
  node: [number, number];
  g: number;
  h: number;
  f: number;
  frontier_size: number;
}

interface AStarResult {
  success: boolean;
  path: number[][];
  path_found: boolean;
  path_length: number;
  nodes_explored: number;
  execution_time_ms: number;
  grid: number[][];
  start: [number, number];
  goal: [number, number];
  step_trace: StepTrace[];
}

function AStarDemo() {
  const [parameters, setParameters] = useState<AStarParameters>({
    grid_size: 15,
    obstacle_density: 0.25,
    heuristic: 'manhattan',
    random_state: 42,
  });

  const [result, setResult] = useState<AStarResult | null>(null);

  const trainMutation = useMutation({
    mutationFn: (params: AStarParameters) =>
      apiService.trainAlgorithm('classical-ai', 'astar', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof AStarParameters, value: number | string | boolean) => {
    setParameters((prev) => ({ ...prev, [name]: value }));
  };

  const steps = result?.step_trace ?? [];
  const { currentStep, playerProps } = useStepPlayback(steps.length);
  const currentTrace = steps[currentStep];

  return (
    <AlgorithmLayout
      title="A* Pathfinding"
      description="Informed search algorithm that finds the shortest path using heuristic guidance"
      category="Classical AI"
      difficulty="Intermediate"
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Parameters</CardTitle>
              <CardDescription>Configure the grid and search</CardDescription>
            </CardHeader>
            <CardContent>
              <Controls
                parameters={parameters}
                onParameterChange={handleParameterChange as (name: string, value: number | string | boolean) => void}
                onTrain={handleTrain}
                isTraining={trainMutation.isPending}
              />
            </CardContent>
          </Card>

          {result && (
            <Card>
              <CardHeader>
                <CardTitle>Results</CardTitle>
                <CardDescription>Search metrics</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Path Length</p>
                    <p className="text-2xl font-bold">{result.path_length}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Nodes Explored</p>
                    <p className="text-2xl font-bold">{result.nodes_explored}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Execution Time</p>
                    <p className="text-2xl font-bold">{result.execution_time_ms.toFixed(2)}ms</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Status</p>
                    <p className="text-2xl font-bold">
                      {result.path_found ? '✓ Found' : '✗ Not Found'}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        <div className="space-y-6">
          {trainMutation.isPending && (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <div className="text-center space-y-4">
                  <Loader2 className="h-8 w-8 animate-spin mx-auto text-primary" />
                  <p className="text-muted-foreground">Finding path with A*...</p>
                </div>
              </CardContent>
            </Card>
          )}

          {trainMutation.isError && (
            <Card className="border-destructive">
              <CardHeader>
                <CardTitle className="text-destructive">Search Failed</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-destructive">
                  {(trainMutation.error as Error)?.message ||
                    'An error occurred during pathfinding'}
                </p>
              </CardContent>
            </Card>
          )}

          {result && (
            <Visualization
              grid={result.grid}
              start={result.start}
              goal={result.goal}
              path={result.path}
              pathFound={result.path_found}
              highlightedNode={currentTrace?.node}
            />
          )}

          {steps.length > 1 && (
            <>
              <StepPlayer totalSteps={steps.length} {...playerProps} />
              <MathPanel
                title="A* Cost Function"
                formula="f(n) = g(n) + h(n)"
                substitution={
                  currentTrace
                    ? `g(n) = ${currentTrace.g.toFixed(1)}, h(n) = ${currentTrace.h.toFixed(1)}, f(n) = ${currentTrace.f.toFixed(1)}`
                    : undefined
                }
                note="A* expands the node with the lowest f(n); g(n) is the actual cost so far, h(n) is the heuristic estimate to the goal."
              />
            </>
          )}
        </div>
      </div>

      <Documentation />
    </AlgorithmLayout>
  );
}

export default AStarDemo;
