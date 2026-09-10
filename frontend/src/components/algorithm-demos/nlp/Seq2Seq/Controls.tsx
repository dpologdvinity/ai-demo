import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';

interface Seq2SeqParams {
  task: string;
  hidden_size: number;
  num_layers: number;
  dropout: number;
  attention: boolean;
  epochs: number;
  learning_rate: number;
  teacher_forcing_ratio: number;
  language_pair: string;
  use_custom_pairs: boolean;
  custom_input_sequences: string[];
  custom_target_sequences: string[];
}

interface ControlsProps {
  parameters: Seq2SeqParams;
  onChange: (name: keyof Seq2SeqParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  const handleCustomInputChange = (value: string) => {
    const sequences = value.split('\n').filter(line => line.trim());
    onChange('custom_input_sequences', sequences);
  };

  const handleCustomTargetChange = (value: string) => {
    const sequences = value.split('\n').filter(line => line.trim());
    onChange('custom_target_sequences', sequences);
  };

  return (
    <div className="space-y-4">
      {/* Task Selection */}
      <div className="space-y-2">
        <Label>Task Type</Label>
        <Select
          value={parameters.task}
          onValueChange={(value) => onChange('task', value)}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="translation">Machine Translation</SelectItem>
            <SelectItem value="reversal">Sequence Reversal</SelectItem>
            <SelectItem value="date-conversion">Date Format Conversion</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Type of sequence transformation task
        </p>
      </div>

      {/* Language Pair (only for translation) */}
      {parameters.task === 'translation' && (
        <div className="space-y-2">
          <Label>Language Pair</Label>
          <Select
            value={parameters.language_pair}
            onValueChange={(value) => onChange('language_pair', value)}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="en-fr">English-French</SelectItem>
              <SelectItem value="en-es">English-Spanish</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Source and target languages
          </p>
        </div>
      )}

      {/* Hidden Size */}
      <ParameterControl
        label="Hidden Size"
        description="Number of units in LSTM layers (64-512)"
        type="range"
        value={parameters.hidden_size}
        onChange={(value) => onChange('hidden_size', parseInt(String(value)))}
        min={64}
        max={512}
        step={64}
      />

      {/* Number of Layers */}
      <ParameterControl
        label="Number of Layers"
        description="LSTM layers in encoder/decoder"
        type="range"
        value={parameters.num_layers}
        onChange={(value) => onChange('num_layers', parseInt(String(value)))}
        min={1}
        max={4}
        step={1}
      />

      {/* Dropout */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Dropout Rate</Label>
          <span className="text-sm text-muted-foreground">
            {parameters.dropout.toFixed(2)}
          </span>
        </div>
        <Slider
          value={[parameters.dropout]}
          onValueChange={([value]) => onChange('dropout', value)}
          min={0}
          max={0.5}
          step={0.05}
        />
        <p className="text-xs text-muted-foreground">
          Regularization to prevent overfitting
        </p>
      </div>

      {/* Attention */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Use Attention Mechanism</Label>
          <Switch
            checked={parameters.attention}
            onCheckedChange={(checked) => onChange('attention', checked)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Enable Bahdanau attention for better long sequences
        </p>
      </div>

      {/* Learning Rate */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Learning Rate</Label>
          <span className="text-sm text-muted-foreground">
            {parameters.learning_rate.toFixed(4)}
          </span>
        </div>
        <Slider
          value={[parameters.learning_rate]}
          onValueChange={([value]) => onChange('learning_rate', value)}
          min={0.0001}
          max={0.01}
          step={0.0001}
        />
        <p className="text-xs text-muted-foreground">
          Adam optimizer learning rate
        </p>
      </div>

      {/* Epochs */}
      <ParameterControl
        label="Training Epochs"
        description="Number of training iterations"
        type="range"
        value={parameters.epochs}
        onChange={(value) => onChange('epochs', parseInt(String(value)))}
        min={10}
        max={200}
        step={10}
      />

      {/* Teacher Forcing Ratio */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Teacher Forcing Ratio</Label>
          <span className="text-sm text-muted-foreground">
            {parameters.teacher_forcing_ratio.toFixed(2)}
          </span>
        </div>
        <Slider
          value={[parameters.teacher_forcing_ratio]}
          onValueChange={([value]) => onChange('teacher_forcing_ratio', value)}
          min={0}
          max={1}
          step={0.1}
        />
        <p className="text-xs text-muted-foreground">
          Probability of using ground truth during training
        </p>
      </div>

      {/* Use Custom Pairs */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Use Custom Sequence Pairs</Label>
          <Switch
            checked={parameters.use_custom_pairs}
            onCheckedChange={(checked) => onChange('use_custom_pairs', checked)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Train on your own input-target sequence pairs
        </p>
      </div>

      {parameters.use_custom_pairs && (
        <>
          <div className="space-y-2">
            <Label>Input Sequences (one per line)</Label>
            <Textarea
              value={parameters.custom_input_sequences.join('\n')}
              onChange={(e) => handleCustomInputChange(e.target.value)}
              placeholder="Enter source sequences..."
              rows={6}
              className="font-mono text-sm"
            />
            <p className="text-xs text-muted-foreground">
              {parameters.custom_input_sequences.length} input(s) entered
            </p>
          </div>

          <div className="space-y-2">
            <Label>Target Sequences (one per line)</Label>
            <Textarea
              value={parameters.custom_target_sequences.join('\n')}
              onChange={(e) => handleCustomTargetChange(e.target.value)}
              placeholder="Enter target sequences..."
              rows={6}
              className="font-mono text-sm"
            />
            <p className="text-xs text-muted-foreground">
              {parameters.custom_target_sequences.length} target(s) entered
            </p>
          </div>
        </>
      )}

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Architecture:</strong> LSTM Encoder-Decoder with optional attention
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          Attention mechanism helps model focus on relevant input for each output
        </p>
      </div>
    </div>
  );
}
