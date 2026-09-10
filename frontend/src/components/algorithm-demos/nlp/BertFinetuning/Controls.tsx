import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';

interface BertParams {
  model_name: string;
  learning_rate: number;
  epochs: number;
  batch_size: number;
  max_length: number;
  use_custom_dataset: boolean;
  custom_texts: string[];
  custom_labels: number[];
}

interface ControlsProps {
  parameters: BertParams;
  onChange: (name: keyof BertParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  const handleCustomTextsChange = (value: string) => {
    const texts = value.split('\n').filter(line => line.trim());
    onChange('custom_texts', texts);
  };

  const handleCustomLabelsChange = (value: string) => {
    const labels = value
      .split('\n')
      .filter(line => line.trim())
      .map(line => parseInt(line.trim()))
      .filter(n => !isNaN(n));
    onChange('custom_labels', labels);
  };

  return (
    <div className="space-y-4">
      {/* Model Selection */}
      <div className="space-y-2">
        <Label>Model Variant</Label>
        <Select
          value={parameters.model_name}
          onValueChange={(value) => onChange('model_name', value)}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="bert-base-uncased">BERT Base Uncased</SelectItem>
            <SelectItem value="distilbert-base-uncased">DistilBERT Base Uncased</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Pre-trained BERT variant to fine-tune
        </p>
      </div>

      {/* Learning Rate */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Learning Rate</Label>
          <span className="text-sm text-muted-foreground">
            {parameters.learning_rate.toExponential(2)}
          </span>
        </div>
        <Slider
          value={[parameters.learning_rate]}
          onValueChange={([value]) => onChange('learning_rate', value)}
          min={1e-5}
          max={5e-5}
          step={5e-6}
        />
        <p className="text-xs text-muted-foreground">
          Smaller learning rates are more stable for fine-tuning
        </p>
      </div>

      {/* Epochs */}
      <ParameterControl
        label="Training Epochs"
        description="Number of passes through the training data"
        type="range"
        value={parameters.epochs}
        onChange={(value) => onChange('epochs', parseInt(String(value)))}
        min={1}
        max={10}
        step={1}
      />

      {/* Batch Size */}
      <ParameterControl
        label="Batch Size"
        description="Number of samples per training step"
        type="select"
        value={parameters.batch_size}
        onChange={(value) => onChange('batch_size', value)}
        options={[
          { label: '8', value: 8 },
          { label: '16', value: 16 },
          { label: '32', value: 32 },
        ]}
      />

      {/* Max Length */}
      <ParameterControl
        label="Max Sequence Length"
        description="Maximum tokens per text sample (64-512)"
        type="range"
        value={parameters.max_length}
        onChange={(value) => onChange('max_length', parseInt(String(value)))}
        min={64}
        max={512}
        step={64}
      />

      {/* Use Custom Dataset */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Use Custom Dataset</Label>
          <Switch
            checked={parameters.use_custom_dataset}
            onCheckedChange={(checked) => onChange('use_custom_dataset', checked)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Fine-tune on your own text samples with labels
        </p>
      </div>

      {parameters.use_custom_dataset && (
        <>
          <div className="space-y-2">
            <Label>Custom Texts (one per line)</Label>
            <Textarea
              value={parameters.custom_texts.join('\n')}
              onChange={(e) => handleCustomTextsChange(e.target.value)}
              placeholder="Enter texts here, one per line..."
              rows={6}
              className="font-mono text-sm"
            />
            <p className="text-xs text-muted-foreground">
              {parameters.custom_texts.length} text(s) entered
            </p>
          </div>

          <div className="space-y-2">
            <Label>Custom Labels (0/1/2, one per line)</Label>
            <Textarea
              value={parameters.custom_labels.join('\n')}
              onChange={(e) => handleCustomLabelsChange(e.target.value)}
              placeholder="Enter labels: 0 (negative), 1 (neutral), 2 (positive)..."
              rows={6}
              className="font-mono text-sm"
            />
            <p className="text-xs text-muted-foreground">
              {parameters.custom_labels.length} label(s) entered
            </p>
          </div>
        </>
      )}

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Default Dataset:</strong> Movie reviews (3-class sentiment classification)
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Classes: 0 = Negative, 1 = Neutral, 2 = Positive
        </p>
      </div>
    </div>
  );
}
