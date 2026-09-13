import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';
import Button from '@/components/common/Button';
import { ParameterControl } from '@/components/common/ParameterControl';
import { Play } from 'lucide-react';

interface ControlsProps {
  parameters: {
    population_size: number;
    generations: number;
    mutation_rate: number;
    crossover_rate: number;
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
        <CardDescription>Configure genetic algorithm</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <ParameterControl
          label="Population Size"
          type="slider"
          value={parameters.population_size}
          min={10}
          max={200}
          step={10}
          onChange={(value) => onParameterChange('population_size', value as number)}
          description="Number of individuals per generation"
        />

        <ParameterControl
          label="Generations"
          type="slider"
          value={parameters.generations}
          min={5}
          max={500}
          step={5}
          onChange={(value) => onParameterChange('generations', value as number)}
          description="Number of generations to evolve"
        />

        <ParameterControl
          label="Mutation Rate"
          type="slider"
          value={parameters.mutation_rate}
          min={0}
          max={1}
          step={0.05}
          onChange={(value) => onParameterChange('mutation_rate', value as number)}
          description="Probability of random gene mutation (0-1)"
        />

        <ParameterControl
          label="Crossover Rate"
          type="slider"
          value={parameters.crossover_rate}
          min={0}
          max={1}
          step={0.05}
          onChange={(value) => onParameterChange('crossover_rate', value as number)}
          description="Probability of genetic recombination (0-1)"
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
