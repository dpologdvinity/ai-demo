import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';
import Button from '@/components/common/Button';
import { ParameterControl } from '@/components/common/ParameterControl';
import { Play } from 'lucide-react';

interface ControlsProps {
  parameters: {
    initial_temperature: number;
    cooling_rate: number;
    max_iterations: number;
    random_state: number;
  };
  onParameterChange: (name: string, value: number) => void;
  onRun: () => void;
  isRunning: boolean;
}

function Controls({ parameters, onParameterChange, onRun, isRunning }: ControlsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Parameters</CardTitle>
        <CardDescription>Configure simulated annealing</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <ParameterControl
          label="Initial Temperature"
          type="slider"
          value={parameters.initial_temperature}
          min={0.1}
          max={100}
          step={0.5}
          onChange={(value) => onParameterChange('initial_temperature', value as number)}
          description="Starting temperature controls exploration (0.1-100)"
        />

        <ParameterControl
          label="Cooling Rate"
          type="slider"
          value={parameters.cooling_rate}
          min={0.8}
          max={0.999}
          step={0.005}
          onChange={(value) => onParameterChange('cooling_rate', value as number)}
          description="Temperature decay rate (0.8-0.999)"
        />

        <ParameterControl
          label="Max Iterations"
          type="slider"
          value={parameters.max_iterations}
          min={20}
          max={1000}
          step={10}
          onChange={(value) => onParameterChange('max_iterations', value as number)}
          description="Maximum iterations (20-1000)"
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
          onClick={onRun}
          disabled={isRunning}
          className="w-full"
        >
          <Play className="h-4 w-4 mr-2" />
          {isRunning ? 'Running...' : 'Run Algorithm'}
        </Button>
      </CardContent>
    </Card>
  );
}

export default Controls;
