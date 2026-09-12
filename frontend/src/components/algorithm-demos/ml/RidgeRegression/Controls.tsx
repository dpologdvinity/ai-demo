import { ParameterControl } from '@/components/common/ParameterControl';

interface ControlsProps {
  parameters: {
    alpha: number;
    fit_intercept: boolean;
    solver: string;
  };
  onChange: (name: string, value: any) => void;
  disabled?: boolean;
}

export function Controls({ parameters, onChange, disabled }: ControlsProps) {
  return (
    <div className="space-y-4">
      <ParameterControl
        label="Alpha (Regularization Strength)"
        value={parameters.alpha}
        onChange={(value) => onChange('alpha', value)}
        type="slider"
        min={0.01}
        max={100.0}
        step={0.01}
        description="Regularization strength. Larger values specify stronger regularization."
        disabled={disabled}
      />

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
        disabled={disabled}
      />

      <ParameterControl
        label="Solver"
        value={parameters.solver}
        onChange={(value) => onChange('solver', value)}
        type="select"
        options={[
          { value: 'auto', label: 'Auto' },
          { value: 'svd', label: 'SVD' },
          { value: 'cholesky', label: 'Cholesky' },
          { value: 'lsqr', label: 'LSQR' },
        ]}
        description="Solver to use in the computational routines"
        disabled={disabled}
      />
    </div>
  );
}
