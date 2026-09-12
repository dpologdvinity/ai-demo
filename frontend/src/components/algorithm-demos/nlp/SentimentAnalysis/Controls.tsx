import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';

interface SentimentAnalysisParams {
  model_type: string;
  confidence_threshold: number;
  neutral_threshold: number;
  use_custom_texts: boolean;
  custom_texts: string[];
}

interface ControlsProps {
  parameters: SentimentAnalysisParams;
  onChange: (name: keyof SentimentAnalysisParams, value: any) => void;
  disabled?: boolean;
}

export function Controls({ parameters, onChange, disabled }: ControlsProps) {
  const handleCustomTextsChange = (value: string) => {
    // Split by newlines and filter empty lines
    const texts = value.split('\n').filter(line => line.trim());
    onChange('custom_texts', texts);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Parameters</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label>Model Type</Label>
          <Select
            value={parameters.model_type}
            onValueChange={(value) => onChange('model_type', value)}
            disabled={disabled}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="vader">VADER (Rule-based)</SelectItem>
              <SelectItem value="textblob">TextBlob</SelectItem>
              <SelectItem value="transformers">Transformers</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Sentiment analysis model to use
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>Confidence Threshold</Label>
            <span className="text-sm text-muted-foreground">
              {parameters.confidence_threshold.toFixed(2)}
            </span>
          </div>
          <Slider
            value={[parameters.confidence_threshold]}
            onValueChange={([value]) => onChange('confidence_threshold', value)}
            min={0}
            max={1}
            step={0.05}
            disabled={disabled}
          />
          <p className="text-xs text-muted-foreground">
            Minimum confidence score threshold
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>Neutral Threshold</Label>
            <span className="text-sm text-muted-foreground">
              {parameters.neutral_threshold.toFixed(3)}
            </span>
          </div>
          <Slider
            value={[parameters.neutral_threshold]}
            onValueChange={([value]) => onChange('neutral_threshold', value)}
            min={0}
            max={0.5}
            step={0.01}
            disabled={disabled}
          />
          <p className="text-xs text-muted-foreground">
            Threshold for neutral classification (VADER compound score range)
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>Use Custom Texts</Label>
            <Switch
              checked={parameters.use_custom_texts}
              onCheckedChange={(checked) => onChange('use_custom_texts', checked)}
              disabled={disabled}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Analyze your own text samples instead of default examples
          </p>
        </div>

        {parameters.use_custom_texts && (
          <div className="space-y-2">
            <Label>Custom Texts (one per line)</Label>
            <Textarea
              value={parameters.custom_texts.join('\n')}
              onChange={(e) => handleCustomTextsChange(e.target.value)}
              placeholder="Enter your texts here, one per line..."
              rows={8}
              disabled={disabled}
              className="font-mono text-sm"
            />
            <p className="text-xs text-muted-foreground">
              {parameters.custom_texts.length} text(s) entered
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
