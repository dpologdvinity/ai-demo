import { ParameterControl } from '@/components/common/ParameterControl';

interface ControlsProps {
  parameters: {
    n_neighbors: number;
    weights: 'uniform' | 'distance';
    metric: 'euclidean' | 'manhattan' | 'minkowski';
    p: number;
    test_size: number;
    random_state: number;
  };
  onChange: (name: string, value: any) => void;
  disabled?: boolean;
}

export function Controls({ parameters, onChange, disabled }: ControlsProps) {
  return (
    <div className="space-y-4">
      <ParameterControl
        label="Number of Neighbors (K)"
        value={parameters.n_neighbors}
        onChange={(value) => onChange('n_neighbors', value)}
        type="slider"
        min={1}
        max={20}
        step={1}
        description="Number of nearest neighbors to consider for classification"
        disabled={disabled}
      />

      <ParameterControl
        label="Weight Function"
        value={parameters.weights}
        onChange={(value) => onChange('weights', value)}
        type="select"
        options={[
          { value: 'uniform', label: 'Uniform (equal weights)' },
          { value: 'distance', label: 'Distance (weighted by inverse distance)' },
        ]}
        description="How to weight neighbors in prediction"
        disabled={disabled}
      />

      <ParameterControl
        label="Distance Metric"
        value={parameters.metric}
        onChange={(value) => onChange('metric', value)}
        type="select"
        options={[
          { value: 'euclidean', label: 'Euclidean' },
          { value: 'manhattan', label: 'Manhattan' },
          { value: 'minkowski', label: 'Minkowski' },
        ]}
        description="Distance metric for finding nearest neighbors"
        disabled={disabled}
      />

      {parameters.metric === 'minkowski' && (
        <ParameterControl
          label="Minkowski Power Parameter (p)"
          value={parameters.p}
          onChange={(value) => onChange('p', value)}
          type="slider"
          min={1}
          max={5}
          step={1}
          description="Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)"
          disabled={disabled}
        />
      )}

      <ParameterControl
        label="Test Size"
        value={parameters.test_size}
        onChange={(value) => onChange('test_size', value)}
        type="slider"
        min={0.1}
        max={0.5}
        step={0.05}
        description="Proportion of data to use for testing"
        disabled={disabled}
      />
    </div>
  );
}
