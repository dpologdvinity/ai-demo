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

interface MinimaxParameters {
  opponent: 'random' | 'optimal';
  ai_starts: boolean;
  use_alpha_beta: boolean;
  random_state: number;
}

interface GameMove {
  move_number: number;
  player: 'ai' | 'opponent';
  position: [number, number];
  board_after: string[][];
}

interface StepTrace {
  move_number: number;
  position: [number, number];
  score: number;
  nodes_evaluated: number;
}

interface MinimaxResult {
  success: boolean;
  winner: 'ai' | 'opponent' | 'draw';
  moves: GameMove[];
  total_nodes_evaluated: number;
  execution_time_ms: number;
  step_trace: StepTrace[];
}

function MinimaxDemo() {
  const [parameters, setParameters] = useState<MinimaxParameters>({
    opponent: 'random',
    ai_starts: true,
    use_alpha_beta: true,
    random_state: 42,
  });

  const [result, setResult] = useState<MinimaxResult | null>(null);

  const trainMutation = useMutation({
    mutationFn: (params: MinimaxParameters) =>
      apiService.trainAlgorithm('classical-ai', 'minimax', params),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleTrain = () => {
    trainMutation.mutate(parameters);
  };

  const handleParameterChange = (name: keyof MinimaxParameters, value: number | string | boolean) => {
    setParameters((prev) => ({ ...prev, [name]: value }));
  };

  const moves = result?.moves ?? [];
  const { currentStep, playerProps } = useStepPlayback(moves.length);
  const currentMove = moves[currentStep];
  const currentBoard = currentMove?.board_after ?? Array(3).fill(null).map(() => Array(3).fill(''));

  // Find the score for the current move from step_trace
  const currentTrace = result?.step_trace?.find((t) => t.move_number === currentStep);

  return (
    <AlgorithmLayout
      title="Minimax with Alpha-Beta Pruning"
      description="Game tree search algorithm for optimal play in two-player games"
      category="Classical AI"
      difficulty="Intermediate"
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Parameters</CardTitle>
              <CardDescription>Configure the game settings</CardDescription>
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
                <CardDescription>Game statistics</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Winner</p>
                    <p className="text-lg font-bold">
                      {result.winner === 'ai' && <span className="text-neon-magenta">AI</span>}
                      {result.winner === 'opponent' && <span className="text-neon-cyan">Opponent</span>}
                      {result.winner === 'draw' && <span className="text-neon-green">Draw</span>}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Total Moves</p>
                    <p className="text-2xl font-bold">{result.moves.length}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Nodes Evaluated</p>
                    <p className="text-2xl font-bold">{result.total_nodes_evaluated}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Execution Time</p>
                    <p className="text-2xl font-bold">{result.execution_time_ms.toFixed(2)}ms</p>
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
                  <p className="text-muted-foreground">Playing tic-tac-toe with minimax...</p>
                </div>
              </CardContent>
            </Card>
          )}

          {trainMutation.isError && (
            <Card className="border-destructive">
              <CardHeader>
                <CardTitle className="text-destructive">Game Failed</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-destructive">
                  {(trainMutation.error as Error)?.message ||
                    'An error occurred during the game'}
                </p>
              </CardContent>
            </Card>
          )}

          {result && (
            <Visualization board={currentBoard} winner={result.winner} />
          )}

          {moves.length > 0 && (
            <>
              <StepPlayer totalSteps={moves.length} {...playerProps} />
              <MathPanel
                title="Minimax Score Function"
                formula="\\text{minimax}(n) = \\begin{cases} \\max_{c \\in children(n)} \\text{minimax}(c) & \\text{if MAX} \\\\ \\min_{c \\in children(n)} \\text{minimax}(c) & \\text{if MIN} \\end{cases}"
                substitution={
                  currentTrace
                    ? `\\text{Score} = ${currentTrace.score}, \\text{Nodes} = ${currentTrace.nodes_evaluated}`
                    : undefined
                }
                note="Each node evaluates the best possible move assuming both players play optimally. Alpha-beta pruning eliminates branches that cannot affect the outcome."
              />
            </>
          )}
        </div>
      </div>

      <Documentation />
    </AlgorithmLayout>
  );
}

export default MinimaxDemo;
