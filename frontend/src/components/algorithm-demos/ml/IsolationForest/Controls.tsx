import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { InfoIcon } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface IsolationForestParams {
  n_estimators: number;
  contamination: number;
  max_samples: string | number;
  random_state: number;
  n_samples: number;
  n_outliers_ratio: number;
}

interface ControlsProps {
  parameters: IsolationForestParams;
  onChange: (name: keyof IsolationForestParams, value: any) => void;
  disabled?: boolean;
}

export function Controls({ parameters, onChange, disabled = false }: ControlsProps) {
  const ControlLabel = ({
    label,
    tooltip,
  }: {
    label: string;
    tooltip?: string;
  }) => (
    <div className="flex items-center gap-2">
      <Label>{label}</Label>
      {tooltip && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <InfoIcon className="h-4 w-4 text-muted-foreground cursor-help" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              <p>{tooltip}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Number of Estimators */}
      <div className="space-y-2">
        <ControlLabel
          label={`Number of Trees: ${parameters.n_estimators}`}
          tooltip="Number of isolation trees in the forest. More trees generally improve detection but increase computation time."
        />
        <Slider
          value={[parameters.n_estimators]}
          onValueChange={([value]) => onChange('n_estimators', value)}
          min={50}
          max={300}
          step={10}
          disabled={disabled}
        />
        <p className="text-xs text-muted-foreground">
          Range: 50-300
        </p>
      </div>

      {/* Contamination */}
      <div className="space-y-2">
        <ControlLabel
          label={`Contamination: ${(parameters.contamination * 100).toFixed(1)}%`}
          tooltip="Expected proportion of outliers in the dataset. This threshold determines how many points are classified as anomalies."
        />
        <Slider
          value={[parameters.contamination * 100]}
          onValueChange={([value]) => onChange('contamination', value / 100)}
          min={1}
          max={50}
          step={1}
          disabled={disabled}
        />
        <p className="text-xs text-muted-foreground">
          Range: 1%-50%
        </p>
      </div>

      {/* Max Samples */}
      <div className="space-y-2">
        <ControlLabel
          label="Max Samples per Tree"
          tooltip="Number of samples to draw for training each tree. 'auto' uses min(256, n_samples). Lower values speed up training."
        />
        <Select
          value={String(parameters.max_samples)}
          onValueChange={(value) =>
            onChange('max_samples', value === 'auto' ? 'auto' : parseInt(value))
          }
          disabled={disabled}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="auto">Auto</SelectItem>
            <SelectItem value="100">100</SelectItem>
            <SelectItem value="256">256</SelectItem>
            <SelectItem value="512">512</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Number of Samples */}
      <div className="space-y-2">
        <ControlLabel
          label={`Number of Data Points: ${parameters.n_samples}`}
          tooltip="Total number of data points to generate for the demonstration."
        />
        <Slider
          value={[parameters.n_samples]}
          onValueChange={([value]) => onChange('n_samples', value)}
          min={100}
          max={1000}
          step={50}
          disabled={disabled}
        />
        <p className="text-xs text-muted-foreground">
          Range: 100-1000
        </p>
      </div>

      {/* Outlier Injection Ratio */}
      <div className="space-y-2">
        <ControlLabel
          label={`Outlier Injection Ratio: ${(parameters.n_outliers_ratio * 100).toFixed(1)}%`}
          tooltip="Ratio of synthetic outliers to inject into the dataset for demonstration. This creates ground truth for evaluation."
        />
        <Slider
          value={[parameters.n_outliers_ratio * 100]}
          onValueChange={([value]) => onChange('n_outliers_ratio', value / 100)}
          min={1}
          max={30}
          step={1}
          disabled={disabled}
        />
        <p className="text-xs text-muted-foreground">
          Range: 1%-30%
        </p>
      </div>

      {/* Random State */}
      <div className="space-y-2">
        <ControlLabel
          label="Random Seed"
          tooltip="Seed for random number generation. Use the same seed for reproducible results."
        />
        <Input
          type="number"
          value={parameters.random_state}
          onChange={(e) =>
            onChange('random_state', parseInt(e.target.value) || 42)
          }
          min={0}
          max={100}
          disabled={disabled}
        />
      </div>
    </div>
  );
}
