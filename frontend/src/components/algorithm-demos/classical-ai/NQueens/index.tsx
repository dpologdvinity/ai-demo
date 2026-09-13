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

interface NQueensParameters {
  board_size: number;
  random_state: number;
}

interface StepTrace {
  step: number;
  column: number;
  row_tried: number;
  accepted: boolean;
  board_state: number[];
}

interface NQueensResult {
  success: boolean;
  solved: boolean;
  solution: number[];
  backtrack_count: number;
  execution_time_ms: number;
  board_size: number;
  step_trace: StepTrace[];
  error?: string;
}

function NQueensDemo() {
  const [parameters, setParameters] = useState<NQueensParameters>({
    board_size: 8,
    random_state: 42,
  });

  const [result, setResult] = useState<NQueensResult | null>(null);

  const solveMutation = useMutation({
    mutationFn: (params: NQueensParameters) =>
      apiService.trainAlgorithm('classical-ai', 'nqueens', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleSolve = () => {
    solveMutation.mutate(parameters);
  };

  const handleParameterChange = (name: string, value: number) => {
    setParameters((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const traceSteps = result?.step_trace ?? [];
  const { currentStep, playerProps } = useStepPlayback(traceSteps.length);
  const currentTrace = traceSteps[currentStep];

  const boardState = currentTrace?.board_state ?? Array(parameters.board_size).fill(-1);

  return (
    <AlgorithmLayout
      title="N-Queens Problem"
      description="Solve the N-Queens constraint satisfaction problem using backtracking"
      category="Classical AI"
      difficulty="Intermediate"
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <Controls
            parameters={parameters}
            onParameterChange={handleParameterChange}
            onSolve={handleSolve}
            isSolving={solveMutation.isPending}
          />

          {result && (
            <Card>
              <CardHeader>
                <CardTitle>Results</CardTitle>
                <CardDescription>Solution statistics</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Status</p>
                    <p className="text-lg font-bold text-green-600 dark:text-green-400">
                      {result.solved ? 'Solved' : 'No Solution'}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Board Size</p>
                    <p className="text-lg font-bold">{result.board_size}×{result.board_size}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Backtrack Count</p>
                    <p className="text-lg font-bold">{result.backtrack_count}</p>
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
          {solveMutation.isPending && (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <div className="text-center space-y-4">
                  <Loader2 className="h-8 w-8 animate-spin mx-auto text-primary" />
                  <p className="text-muted-foreground">Solving N-Queens puzzle...</p>
                </div>
              </CardContent>
            </Card>
          )}

          {solveMutation.isError && (
            <Card className="border-destructive">
              <CardHeader>
                <CardTitle className="text-destructive">Error</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-destructive">
                  {(solveMutation.error as Error)?.message || 'An error occurred'}
                </p>
              </CardContent>
            </Card>
          )}

          {result && (
            <>
              <Visualization
                boardSize={result.board_size}
                boardState={boardState}
                currentColumn={currentTrace?.column}
                currentRowTried={currentTrace?.row_tried}
                accepted={currentTrace?.accepted}
              />

              {traceSteps.length > 1 && (
                <>
                  <StepPlayer totalSteps={traceSteps.length} {...playerProps} />
                  <MathPanel
                    formula="\\forall i \\neq j: \\; row_i \\neq row_j \\;\\wedge\\; |row_i - row_j| \\neq |i - j|"
                    note="No two queens may share a row or diagonal (columns never repeat since each queen occupies its own column)."
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

export default NQueensDemo;
