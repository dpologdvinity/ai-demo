import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';
import Button from '@/components/common/Button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { PlayCircle } from 'lucide-react';

interface GMMParameters {
  n_components: number;
  covariance_type: 'full' | 'tied' | 'diag' | 'spherical';
  max_iter: number;
  n_samples: number;
  random_state: number;
}

interface ControlsProps {
  parameters: GMMParameters;
  onParameterChange: (name: keyof GMMParameters, value: number | string) => void;
  onTrain: () => void;
  isTraining: boolean;
}

function Controls({ parameters, onParameterChange, onTrain, isTraining }: ControlsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Parameters</CardTitle>
        <CardDescription>Configure the Gaussian Mixture Model</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <Label htmlFor="n_components">Number of Components (K)</Label>
            <span className="text-sm font-medium">{parameters.n_components}</span>
          </div>
          <Slider
            id="n_components"
            min={2}
            max={10}
            step={1}
            value={[parameters.n_components]}
            onValueChange={(value) => onParameterChange('n_components', value[0])}
            disabled={isTraining}
          />
          <p className="text-xs text-muted-foreground">
            Number of Gaussian components (clusters) to fit
          </p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="covariance_type">Covariance Type</Label>
          <Select
            value={parameters.covariance_type}
            onValueChange={(value) => onParameterChange('covariance_type', value)}
            disabled={isTraining}
          >
            <SelectTrigger id="covariance_type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="full">Full - Each component has its own covariance</SelectItem>
              <SelectItem value="tied">Tied - All components share covariance</SelectItem>
              <SelectItem value="diag">Diagonal - Only diagonal elements</SelectItem>
              <SelectItem value="spherical">Spherical - Single variance per component</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Type of covariance parameters to use
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <Label htmlFor="max_iter">Maximum Iterations</Label>
            <span className="text-sm font-medium">{parameters.max_iter}</span>
          </div>
          <Slider
            id="max_iter"
            min={10}
            max={500}
            step={10}
            value={[parameters.max_iter]}
            onValueChange={(value) => onParameterChange('max_iter', value[0])}
            disabled={isTraining}
          />
          <p className="text-xs text-muted-foreground">
            Maximum number of EM algorithm iterations
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <Label htmlFor="n_samples">Number of Samples</Label>
            <span className="text-sm font-medium">{parameters.n_samples}</span>
          </div>
          <Slider
            id="n_samples"
            min={100}
            max={1000}
            step={50}
            value={[parameters.n_samples]}
            onValueChange={(value) => onParameterChange('n_samples', value[0])}
            disabled={isTraining}
          />
          <p className="text-xs text-muted-foreground">
            Number of samples in the generated dataset
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <Label htmlFor="random_state">Random Seed</Label>
            <span className="text-sm font-medium">{parameters.random_state}</span>
          </div>
          <Slider
            id="random_state"
            min={0}
            max={100}
            step={1}
            value={[parameters.random_state]}
            onValueChange={(value) => onParameterChange('random_state', value[0])}
            disabled={isTraining}
          />
          <p className="text-xs text-muted-foreground">
            Random seed for reproducibility
          </p>
        </div>

        <Button
          onClick={onTrain}
          disabled={isTraining}
          className="w-full"
          size="lg"
        >
          {isTraining ? (
            <>Training...</>
          ) : (
            <>
              <PlayCircle className="mr-2 h-5 w-5" />
              Train Model
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
}

export default Controls;
