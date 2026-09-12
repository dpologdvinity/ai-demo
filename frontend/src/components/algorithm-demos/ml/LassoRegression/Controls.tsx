import { ParameterControl } from '@/components/common/ParameterControl';

interface ControlsProps {
  parameters: {
    alpha: number;
    max_iter: number;
    selection: string;
    normalize: boolean;
    dataset_name: string;
    test_size: number;
  };
  onChange: (name: string, value: any) => void;
  disabled?: boolean;
}

export function Controls({ parameters, onChange, disabled }: ControlsProps) {
  return (
    <div className="space-y-4">
      <ParameterControl
        label="Regularization Strength (α)"
        value={parameters.alpha}
        onChange={(value) => onChange('alpha', value)}
        type="slider"
        min={0.01}
        max={10.0}
        step={0.1}
        description="L1 penalty strength. Higher values create sparser models with more zero coefficients."
        disabled={disabled}
      />

      <ParameterControl
        label="Maximum Iterations"
        value={parameters.max_iter}
        onChange={(value) => onChange('max_iter', value)}
        type="number"
        min={100}
        max={5000}
        step={100}
        description="Maximum number of iterations for the optimization algorithm."
        disabled={disabled}
      />

      <ParameterControl
        label="Coefficient Update Strategy"
        value={parameters.selection}
        onChange={(value) => onChange('selection', value)}
        type="select"
        options={[
          { value: 'cyclic', label: 'Cyclic (Sequential)' },
          { value: 'random', label: 'Random (Faster for large datasets)' },
        ]}
        description="Strategy for updating coefficients during optimization."
        disabled={disabled}
      />

      <ParameterControl
        label="Normalize Features"
        value={parameters.normalize ? 'true' : 'false'}
        onChange={(value) => onChange('normalize', value === 'true')}
        type="select"
        options={[
          { value: 'true', label: 'True' },
          { value: 'false', label: 'False' },
        ]}
        description="Whether to normalize features before regression using StandardScaler."
        disabled={disabled}
      />

      <ParameterControl
        label="Dataset"
        value={parameters.dataset_name}
        onChange={(value) => onChange('dataset_name', value)}
        type="select"
        options={[
          { value: 'boston', label: 'California Housing' },
        ]}
        description="Dataset to use for training"
        disabled={disabled}
      />

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
