import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { AlgorithmLayout } from '@/components/common/AlgorithmLayout';
import { apiService } from '@/services/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';
import { StepPlayer, useStepPlayback } from '@/components/common/StepPlayer';
import { MathPanel } from '@/components/common/MathPanel';
import { Loader2 } from 'lucide-react';
import Controls from './Controls';
import Visualization from './Visualization';
import Documentation from './Documentation';

interface SimulatedAnnealingParameters {
  initial_temperature: number;
  cooling_rate: number;
  max_iterations: number;
  random_state: number;
}

interface StepData {
  step: number;
  temperature: number;
  current_solution: number;
  current_cost: number;
  candidate_solution: number;
  candidate_cost: number;
  accepted: boolean;
}

interface SimulatedAnnealingResult {
  success: boolean;
  best_solution: number;
  best_cost: number;
  iterations_run: number;
  execution_time_ms: number;
  step_trace: StepData[];
  error?: string;
}

function SimulatedAnnealingDemo() {
  const [parameters, setParameters] = useState<SimulatedAnnealingParameters>({
    initial_temperature: 10,
    cooling_rate: 0.95,
    max_iterations: 200,
    random_state: 42,
  });

  const [result, setResult] = useState<SimulatedAnnealingResult | null>(null);

  const runMutation = useMutation({
    mutationFn: (params: SimulatedAnnealingParameters) =>
      apiService.trainAlgorithm('classical-ai', 'simulated-annealing', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleRun = () => {
    runMutation.mutate(parameters);
  };

  const handleParameterChange = (name: string, value: number) => {
    setParameters((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const trace = result?.step_trace ?? [];
  const { currentStep, playerProps } = useStepPlayback(trace.length);
  const currentTrace = trace[currentStep] || trace[0];

  return (
    <AlgorithmLayout
      title="Simulated Annealing"
      description="Probabilistic optimization through temperature-controlled exploration"
      category="Classical AI"
      difficulty="Intermediate"
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <Controls
            parameters={parameters}
            onParameterChange={handleParameterChange}
            onRun={handleRun}
            isRunning={runMutation.isPending}
          />

          {result && (
            <Card>
              <CardHeader>
                <CardTitle>Results</CardTitle>
                <CardDescription>Optimization results</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Best Solution (x)</p>
                    <p className="text-lg font-bold">{result.best_solution.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Best Cost</p>
                    <p className="text-lg font-bold">{result.best_cost.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Iterations</p>
                    <p className="text-lg font-bold">{result.iterations_run}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Execution Time</p>
                    <p className="text-lg font-bold">{result.execution_time_ms.toFixed(2)}ms</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          <Documentation />
        </div>

        <div className="space-y-6">
          {runMutation.isPending && (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <div className="text-center space-y-4">
                  <Loader2 className="h-8 w-8 animate-spin mx-auto text-primary" />
                  <p className="text-muted-foreground">Running simulated annealing...</p>
                </div>
              </CardContent>
            </Card>
          )}

          {runMutation.isError && (
            <Card className="border-destructive">
              <CardHeader>
                <CardTitle className="text-destructive">Error</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-destructive">
                  {(runMutation.error as Error)?.message || 'An error occurred'}
                </p>
              </CardContent>
            </Card>
          )}

          {result && (
            <>
              <Visualization
                stepData={trace}
                currentStep={currentStep}
              />

              {trace.length > 1 && (
                <>
                  <StepPlayer totalSteps={trace.length} {...playerProps} />
                  <MathPanel
                    formula="P(\\text{accept}) = e^{-\\Delta E / T}"
                    note="Worse solutions are still accepted sometimes early on (high T), letting the search escape local optima; acceptance becomes stricter as T cools."
                    substitution={`\\text{Step } ${currentTrace?.step ?? 0}: \\; T = ${(currentTrace?.temperature ?? 0).toFixed(4)}, \\; E_{\\text{current}} = ${(currentTrace?.current_cost ?? 0).toFixed(4)}`}
                  />
                </>
              )}
            </>
          )}
        </div>
      </div>
    </AlgorithmLayout>
  );
}

export default SimulatedAnnealingDemo;
