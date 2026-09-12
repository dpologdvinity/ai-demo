import { ParameterControl } from '@/components/common/ParameterControl';

interface ControlsProps {
  parameters: {
    fit_intercept: boolean;
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
        label="Fit Intercept"
        value={parameters.fit_intercept ? 'true' : 'false'}
        onChange={(value) => onChange('fit_intercept', value === 'true')}
        type="select"
        options={[
          { value: 'true', label: 'True' },
          { value: 'false', label: 'False' },
        ]}
        description="Whether to calculate the intercept for the model"
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
        description="Whether to normalize features before regression"
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
      />
    </div>
  );
}
