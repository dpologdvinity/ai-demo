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

interface GeneticAlgorithmParameters {
  population_size: number;
  generations: number;
  mutation_rate: number;
  crossover_rate: number;
  random_state: number;
}

interface GenerationSnapshot {
  generation: number;
  best_fitness: number;
  avg_fitness: number;
  best_individual: number;
  population_sample: number[];
}

interface GeneticAlgorithmResult {
  success: boolean;
  best_individual: number;
  best_fitness: number;
  generations_run: number;
  execution_time_ms: number;
  step_trace: GenerationSnapshot[];
  error?: string;
}

function GeneticAlgorithmDemo() {
  const [parameters, setParameters] = useState<GeneticAlgorithmParameters>({
    population_size: 50,
    generations: 100,
    mutation_rate: 0.1,
    crossover_rate: 0.7,
    random_state: 42,
  });

  const [result, setResult] = useState<GeneticAlgorithmResult | null>(null);

  const runMutation = useMutation({
    mutationFn: (params: GeneticAlgorithmParameters) =>
      apiService.trainAlgorithm('classical-ai', 'genetic-algorithm', params),
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
  const currentGen = trace[currentStep] || trace[0];

  return (
    <AlgorithmLayout
      title="Genetic Algorithm"
      description="Evolutionary optimization using selection, crossover, and mutation"
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
                    <p className="text-lg font-bold">{result.best_individual.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Best Fitness</p>
                    <p className="text-lg font-bold">{result.best_fitness.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Generations</p>
                    <p className="text-lg font-bold">{result.generations_run}</p>
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
                  <p className="text-muted-foreground">Running genetic algorithm...</p>
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
                generationData={trace}
                currentGeneration={currentStep}
              />

              {trace.length > 1 && (
                <>
                  <StepPlayer totalSteps={trace.length} {...playerProps} />
                  <MathPanel
                    title="Selection Probability"
                    formula="p_i = \\frac{f_i}{\\sum_j f_j}"
                    note="Fitter individuals are more likely to be selected as parents each generation."
                    substitution={`Gen ${currentGen?.generation ?? 0}: \\; f_{{\\text{{best}}}} = ${(currentGen?.best_fitness ?? 0).toFixed(4)}, \\; f_{{\\text{{avg}}}} = ${(currentGen?.avg_fitness ?? 0).toFixed(4)}`}
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

export default GeneticAlgorithmDemo;
