import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface ControlsProps {
  parameters: {
    n_clusters: number;
    linkage: string;
    affinity: string;
    n_samples: number;
  };
  onChange: (name: string, value: any) => void;
  disabled?: boolean;
}

export function Controls({ parameters, onChange, disabled = false }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Number of Clusters */}
      <div className="space-y-2">
        <Label htmlFor="n_clusters">Number of Clusters: {parameters.n_clusters}</Label>
        <Input
          id="n_clusters"
          type="range"
          min="2"
          max="10"
          step="1"
          value={parameters.n_clusters}
          onChange={(e) => onChange('n_clusters', parseInt(e.target.value))}
          disabled={disabled}
          className="w-full"
        />
        <p className="text-xs text-muted-foreground">
          Number of clusters to find (2-10)
        </p>
      </div>

      {/* Linkage Method */}
      <div className="space-y-2">
        <Label htmlFor="linkage">Linkage Method</Label>
        <Select
          value={parameters.linkage}
          onValueChange={(value) => onChange('linkage', value)}
          disabled={disabled}
        >
          <SelectTrigger id="linkage">
            <SelectValue placeholder="Select linkage method" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="ward">Ward</SelectItem>
            <SelectItem value="complete">Complete</SelectItem>
            <SelectItem value="average">Average</SelectItem>
            <SelectItem value="single">Single</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          {parameters.linkage === 'ward' && 'Minimizes variance within clusters'}
          {parameters.linkage === 'complete' && 'Maximum distances between all observations'}
          {parameters.linkage === 'average' && 'Average distances between all observations'}
          {parameters.linkage === 'single' && 'Minimum distances between all observations'}
        </p>
      </div>

      {/* Distance Metric */}
      <div className="space-y-2">
        <Label htmlFor="affinity">Distance Metric</Label>
        <Select
          value={parameters.affinity}
          onValueChange={(value) => onChange('affinity', value)}
          disabled={disabled || parameters.linkage === 'ward'}
        >
          <SelectTrigger id="affinity">
            <SelectValue placeholder="Select distance metric" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="euclidean">Euclidean</SelectItem>
            <SelectItem value="manhattan">Manhattan</SelectItem>
            <SelectItem value="cosine">Cosine</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          {parameters.linkage === 'ward'
            ? 'Ward linkage requires Euclidean distance'
            : 'Metric used to compute distances between samples'}
        </p>
      </div>

      {/* Number of Samples */}
      <div className="space-y-2">
        <Label htmlFor="n_samples">Number of Samples: {parameters.n_samples}</Label>
        <Input
          id="n_samples"
          type="range"
          min="50"
          max="1000"
          step="50"
          value={parameters.n_samples}
          onChange={(e) => onChange('n_samples', parseInt(e.target.value))}
          disabled={disabled}
          className="w-full"
        />
        <p className="text-xs text-muted-foreground">
          Number of data points to generate (50-1000)
        </p>
      </div>
    </div>
  );
}
