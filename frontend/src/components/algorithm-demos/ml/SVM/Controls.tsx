import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface SVMParams {
  C: number;
  kernel: string;
  gamma: string;
  degree: number;
}

interface ControlsProps {
  parameters: SVMParams;
  onChange: (name: keyof SVMParams, value: any) => void;
  disabled?: boolean;
}

export function Controls({ parameters, onChange, disabled }: ControlsProps) {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label htmlFor="C">Regularization Parameter (C)</Label>
          <span className="text-sm text-muted-foreground">{parameters.C.toFixed(1)}</span>
        </div>
        <Slider
          id="C"
          min={0.1}
          max={10}
          step={0.1}
          value={[parameters.C]}
          onValueChange={([value]) => onChange('C', value)}
          disabled={disabled}
        />
        <p className="text-xs text-muted-foreground">
          Controls the trade-off between smooth decision boundary and classifying training points correctly.
          Lower values create a smoother boundary.
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="kernel">Kernel Type</Label>
        <Select
          value={parameters.kernel}
          onValueChange={(value) => onChange('kernel', value)}
          disabled={disabled}
        >
          <SelectTrigger id="kernel">
            <SelectValue placeholder="Select kernel" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="linear">Linear</SelectItem>
            <SelectItem value="poly">Polynomial</SelectItem>
            <SelectItem value="rbf">RBF (Radial Basis Function)</SelectItem>
            <SelectItem value="sigmoid">Sigmoid</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Kernel function used to transform data into higher dimensions.
          RBF is good for non-linear patterns, Linear for linearly separable data.
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="gamma">Kernel Coefficient (Gamma)</Label>
        <Select
          value={parameters.gamma}
          onValueChange={(value) => onChange('gamma', value)}
          disabled={disabled}
        >
          <SelectTrigger id="gamma">
            <SelectValue placeholder="Select gamma" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="scale">Scale (1 / (n_features * X.var()))</SelectItem>
            <SelectItem value="auto">Auto (1 / n_features)</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Defines the influence of a single training example.
          Low values mean far, high values mean close influence.
        </p>
      </div>

      {parameters.kernel === 'poly' && (
        <div className="space-y-2">
          <div className="flex justify-between">
            <Label htmlFor="degree">Polynomial Degree</Label>
            <span className="text-sm text-muted-foreground">{parameters.degree}</span>
          </div>
          <Slider
            id="degree"
            min={2}
            max={5}
            step={1}
            value={[parameters.degree]}
            onValueChange={([value]) => onChange('degree', value)}
            disabled={disabled}
          />
          <p className="text-xs text-muted-foreground">
            Degree of the polynomial kernel function. Higher degrees can fit more complex boundaries
            but may overfit.
          </p>
        </div>
      )}
    </div>
  );
}
