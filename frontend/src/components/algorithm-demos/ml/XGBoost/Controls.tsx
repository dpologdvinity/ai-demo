import { ParameterControl } from '@/components/common/ParameterControl';
import Button from '@/components/common/Button';
import { Loader2 } from 'lucide-react';

interface ControlsProps {
  parameters: {
    n_estimators: number;
    learning_rate: number;
    max_depth: number;
    subsample: number;
    dataset_name: string;
    normalize: boolean;
  };
  onParameterChange: (name: string, value: number | string | boolean) => void;
  onTrain: () => void;
  isTraining: boolean;
}

export function Controls({
  parameters,
  onParameterChange,
  onTrain,
  isTraining,
}: ControlsProps) {
  return (
    <div className="space-y-6">
      <ParameterControl
        label="Number of Estimators"
        value={parameters.n_estimators}
        onChange={(value) => onParameterChange('n_estimators', value)}
        min={10}
        max={500}
        step={10}
        type="slider"
        description="Number of boosting rounds (trees to build)"
      />

      <ParameterControl
        label="Learning Rate"
        value={parameters.learning_rate}
        onChange={(value) => onParameterChange('learning_rate', value)}
        min={0.01}
        max={1.0}
        step={0.01}
        type="slider"
        description="Step size shrinkage used to prevent overfitting"
      />

      <ParameterControl
        label="Max Depth"
        value={parameters.max_depth}
        onChange={(value) => onParameterChange('max_depth', value)}
        min={3}
        max={15}
        step={1}
        type="slider"
        description="Maximum tree depth for base learners"
      />

      <ParameterControl
        label="Subsample Ratio"
        value={parameters.subsample}
        onChange={(value) => onParameterChange('subsample', value)}
        min={0.5}
        max={1.0}
        step={0.1}
        type="slider"
        description="Subsample ratio of the training instances"
      />

      <ParameterControl
        label="Dataset"
        value={parameters.dataset_name}
        onChange={(value) => onParameterChange('dataset_name', value)}
        type="select"
        options={[
          { value: 'wine', label: 'Wine Dataset' },
          { value: 'iris', label: 'Iris Dataset' },
          { value: 'digits', label: 'Digits Dataset' },
        ]}
        description="Dataset to use for training"
      />

      <Button
        onClick={onTrain}
        disabled={isTraining}
        className="w-full"
      >
        {isTraining ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Training...
          </>
        ) : (
          'Train Model'
        )}
      </Button>
    </div>
  );
}
