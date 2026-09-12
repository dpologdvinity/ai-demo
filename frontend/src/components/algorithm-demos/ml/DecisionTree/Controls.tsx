import { ParameterControl } from '@/components/common/ParameterControl';

interface ControlsProps {
  parameters: {
    max_depth: number | null;
    min_samples_split: number;
    min_samples_leaf: number;
    criterion: string;
    dataset_name: string;
    random_state: number;
  };
  onChange: (name: string, value: any) => void;
  disabled?: boolean;
}

export function Controls({ parameters, onChange, disabled }: ControlsProps) {
  return (
    <div className="space-y-4">
      <ParameterControl
        label="Maximum Depth"
        value={parameters.max_depth ?? 10}
        onChange={(value) => onChange('max_depth', value === 0 ? null : value)}
        type="slider"
        min={0}
        max={20}
        step={1}
        description="Maximum depth of the tree (0 for unlimited)"
        disabled={disabled}
      />

      <ParameterControl
        label="Min Samples Split"
        value={parameters.min_samples_split}
        onChange={(value) => onChange('min_samples_split', value)}
        type="slider"
        min={2}
        max={20}
        step={1}
        description="Minimum samples required to split an internal node"
        disabled={disabled}
      />

      <ParameterControl
        label="Min Samples Leaf"
        value={parameters.min_samples_leaf}
        onChange={(value) => onChange('min_samples_leaf', value)}
        type="slider"
        min={1}
        max={10}
        step={1}
        description="Minimum samples required at a leaf node"
        disabled={disabled}
      />

      <ParameterControl
        label="Split Criterion"
        value={parameters.criterion}
        onChange={(value) => onChange('criterion', value)}
        type="select"
        options={[
          { value: 'gini', label: 'Gini Impurity' },
          { value: 'entropy', label: 'Entropy (Information Gain)' },
        ]}
        description="Function to measure the quality of a split"
        disabled={disabled}
      />

      <ParameterControl
        label="Dataset"
        value={parameters.dataset_name}
        onChange={(value) => onChange('dataset_name', value)}
        type="select"
        options={[
          { value: 'iris', label: 'Iris Dataset' },
          { value: 'wine', label: 'Wine Dataset' },
          { value: 'digits', label: 'Digits Dataset' },
        ]}
        description="Dataset to use for training"
        disabled={disabled}
      />
    </div>
  );
}
