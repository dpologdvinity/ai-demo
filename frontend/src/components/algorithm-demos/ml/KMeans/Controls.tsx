import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';
import Button from '@/components/common/Button';
import { Play } from 'lucide-react';

interface ControlsProps {
  parameters: {
    n_clusters: number;
    max_iter: number;
    n_init: number;
    random_state: number;
    n_samples: number;
  };
  onParameterChange: (name: string, value: number) => void;
  onTrain: () => void;
  isTraining: boolean;
}

function Controls({ parameters, onParameterChange, onTrain, isTraining }: ControlsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Parameters</CardTitle>
        <CardDescription>Configure K-Means clustering algorithm</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label htmlFor="n_clusters" className="text-sm font-medium">
            Number of Clusters (K)
          </label>
          <input
            id="n_clusters"
            type="number"
            min={2}
            max={10}
            step={1}
            value={parameters.n_clusters}
            onChange={(e) => onParameterChange('n_clusters', parseInt(e.target.value))}
            className="w-full px-3 py-2 border rounded-md"
          />
          <p className="text-xs text-muted-foreground">
            Number of clusters to form (2-10)
          </p>
        </div>

        <div className="space-y-2">
          <label htmlFor="max_iter" className="text-sm font-medium">
            Maximum Iterations
          </label>
          <input
            id="max_iter"
            type="number"
            min={50}
            max={1000}
            step={50}
            value={parameters.max_iter}
            onChange={(e) => onParameterChange('max_iter', parseInt(e.target.value))}
            className="w-full px-3 py-2 border rounded-md"
          />
          <p className="text-xs text-muted-foreground">
            Maximum iterations for convergence (50-1000)
          </p>
        </div>

        <div className="space-y-2">
          <label htmlFor="n_init" className="text-sm font-medium">
            Number of Initializations
          </label>
          <input
            id="n_init"
            type="number"
            min={1}
            max={20}
            step={1}
            value={parameters.n_init}
            onChange={(e) => onParameterChange('n_init', parseInt(e.target.value))}
            className="w-full px-3 py-2 border rounded-md"
          />
          <p className="text-xs text-muted-foreground">
            Number of times to run with different centroid seeds (1-20)
          </p>
        </div>

        <div className="space-y-2">
          <label htmlFor="n_samples" className="text-sm font-medium">
            Number of Samples
          </label>
          <input
            id="n_samples"
            type="number"
            min={100}
            max={1000}
            step={50}
            value={parameters.n_samples}
            onChange={(e) => onParameterChange('n_samples', parseInt(e.target.value))}
            className="w-full px-3 py-2 border rounded-md"
          />
          <p className="text-xs text-muted-foreground">
            Number of data points to generate (100-1000)
          </p>
        </div>

        <div className="space-y-2">
          <label htmlFor="random_state" className="text-sm font-medium">
            Random Seed
          </label>
          <input
            id="random_state"
            type="number"
            min={0}
            max={100}
            step={1}
            value={parameters.random_state}
            onChange={(e) => onParameterChange('random_state', parseInt(e.target.value))}
            className="w-full px-3 py-2 border rounded-md"
          />
          <p className="text-xs text-muted-foreground">
            Random seed for reproducibility (0-100)
          </p>
        </div>

        <Button
          onClick={onTrain}
          disabled={isTraining}
          className="w-full"
        >
          <Play className="h-4 w-4 mr-2" />
          {isTraining ? 'Training...' : 'Train Model'}
        </Button>
      </CardContent>
    </Card>
  );
}

export default Controls;
