import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';

interface ControlsProps {
  parameters: {
    alpha: number;
    l1_ratio: number;
    max_iter: number;
    fit_intercept: boolean;
    normalize: boolean;
  };
  onChange: (name: string, value: any) => void;
  disabled?: boolean;
}

export function Controls({ parameters, onChange, disabled }: ControlsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Model Parameters</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="alpha">
            Regularization Strength (α): {parameters.alpha.toFixed(2)}
          </Label>
          <Input
            id="alpha"
            type="range"
            min={0.01}
            max={10}
            step={0.1}
            value={parameters.alpha}
            onChange={(e) => onChange('alpha', parseFloat(e.target.value))}
            disabled={disabled}
          />
          <p className="text-xs text-muted-foreground">
            Higher values create more regularization and smaller coefficients
          </p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="l1_ratio">
            L1 Ratio (ρ): {parameters.l1_ratio.toFixed(2)}{' '}
            <span className="text-muted-foreground">
              ({parameters.l1_ratio === 0 ? 'Ridge (L2)' :
                parameters.l1_ratio === 1 ? 'Lasso (L1)' :
                'Elastic Net (L1+L2)'})
            </span>
          </Label>
          <Input
            id="l1_ratio"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={parameters.l1_ratio}
            onChange={(e) => onChange('l1_ratio', parseFloat(e.target.value))}
            disabled={disabled}
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>← Ridge (L2 only)</span>
            <span>Elastic Net (L1+L2)</span>
            <span>Lasso (L1 only) →</span>
          </div>
          <p className="text-xs text-muted-foreground">
            Controls the mix between L1 (sparsity) and L2 (stability) regularization
          </p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="max_iter">Maximum Iterations: {parameters.max_iter}</Label>
          <Input
            id="max_iter"
            type="number"
            min={100}
            max={5000}
            step={100}
            value={parameters.max_iter}
            onChange={(e) => onChange('max_iter', parseInt(e.target.value))}
            disabled={disabled}
          />
          <p className="text-xs text-muted-foreground">
            Maximum iterations for the coordinate descent algorithm
          </p>
        </div>

        <div className="flex items-center justify-between">
          <Label htmlFor="fit_intercept">Fit Intercept</Label>
          <Switch
            id="fit_intercept"
            checked={parameters.fit_intercept}
            onCheckedChange={(checked) => onChange('fit_intercept', checked)}
            disabled={disabled}
          />
        </div>

        <div className="flex items-center justify-between">
          <Label htmlFor="normalize">Normalize Features</Label>
          <Switch
            id="normalize"
            checked={parameters.normalize}
            onCheckedChange={(checked) => onChange('normalize', checked)}
            disabled={disabled}
          />
        </div>
      </CardContent>
    </Card>
  );
}
