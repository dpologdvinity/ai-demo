import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';
import Button from '@/components/common/Button';
import { ParameterControl } from '@/components/common/ParameterControl';
import { Play } from 'lucide-react';

interface ControlsProps {
  parameters: {
    board_size: number;
    random_state: number;
  };
  onParameterChange: (name: string, value: number) => void;
  onSolve: () => void;
  isSolving: boolean;
}

function Controls({ parameters, onParameterChange, onSolve, isSolving }: ControlsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Parameters</CardTitle>
        <CardDescription>Configure N-Queens problem</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <ParameterControl
          label="Board Size (N)"
          type="slider"
          value={parameters.board_size}
          min={4}
          max={12}
          step={1}
          onChange={(value) => onParameterChange('board_size', value as number)}
          description="Size of the chessboard (4-12 queens)"
        />

        <ParameterControl
          label="Random Seed"
          type="number"
          value={parameters.random_state}
          min={0}
          max={100}
          step={1}
          onChange={(value) => onParameterChange('random_state', value as number)}
          description="Seed for reproducibility"
        />

        <Button
          onClick={onSolve}
          disabled={isSolving}
          className="w-full"
        >
          <Play className="h-4 w-4 mr-2" />
          {isSolving ? 'Solving...' : 'Solve Puzzle'}
        </Button>
      </CardContent>
    </Card>
  );
}

export default Controls;
