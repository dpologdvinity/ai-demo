import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';
import Button from '@/components/common/Button';
import { Play } from 'lucide-react';

interface ControlsProps {
  parameters: {
    eps: number;
    min_samples: number;
    metric: string;
    random_state: number;
    n_samples: number;
    noise: number;
  };
  onParameterChange: (name: string, value: number | string) => void;
  onTrain: () => void;
  isTraining: boolean;
}

function Controls({ parameters, onParameterChange, onTrain, isTraining }: ControlsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Parameters</CardTitle>
        <CardDescription>Configure DBSCAN clustering algorithm</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label htmlFor="eps" className="text-sm font-medium">
            Epsilon (ε)
          </label>
          <input
            id="eps"
            type="number"
            min={0.1}
            max={2.0}
            step={0.1}
            value={parameters.eps}
            onChange={(e) => onParameterChange('eps', parseFloat(e.target.value))}
            className="w-full px-3 py-2 border rounded-md"
          />
          <p className="text-xs text-muted-foreground">
            Maximum distance between samples in a neighborhood (0.1-2.0)
          </p>
        </div>

        <div className="space-y-2">
          <label htmlFor="min_samples" className="text-sm font-medium">
            Minimum Samples
          </label>
          <input
            id="min_samples"
            type="number"
            min={2}
            max={20}
            step={1}
            value={parameters.min_samples}
            onChange={(e) => onParameterChange('min_samples', parseInt(e.target.value))}
            className="w-full px-3 py-2 border rounded-md"
          />
          <p className="text-xs text-muted-foreground">
            Minimum samples in neighborhood to form core point (2-20)
          </p>
        </div>

        <div className="space-y-2">
          <label htmlFor="metric" className="text-sm font-medium">
            Distance Metric
          </label>
          <select
            id="metric"
            value={parameters.metric}
            onChange={(e) => onParameterChange('metric', e.target.value)}
            className="w-full px-3 py-2 border rounded-md"
          >
            <option value="euclidean">Euclidean</option>
            <option value="manhattan">Manhattan</option>
          </select>
          <p className="text-xs text-muted-foreground">
            Distance metric for neighborhood calculations
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
          <label htmlFor="noise" className="text-sm font-medium">
            Noise Level
          </label>
          <input
            id="noise"
            type="number"
            min={0.0}
            max={0.5}
            step={0.05}
            value={parameters.noise}
            onChange={(e) => onParameterChange('noise', parseFloat(e.target.value))}
            className="w-full px-3 py-2 border rounded-md"
          />
          <p className="text-xs text-muted-foreground">
            Standard deviation of Gaussian noise (0.0-0.5)
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
