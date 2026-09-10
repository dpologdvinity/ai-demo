import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';

interface Word2VecParams {
  vector_size: number;
  window: number;
  min_count: number;
  sg: number;
  epochs: number;
  use_custom_corpus: boolean;
  custom_corpus?: string;
  random_state: number;
}

interface ControlsProps {
  parameters: Word2VecParams;
  onChange: (name: keyof Word2VecParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Vector Size */}
      <ParameterControl
        label="Vector Size"
        description="Dimensionality of word embeddings"
        type="number"
        value={parameters.vector_size}
        onChange={(value) => onChange('vector_size', parseInt(String(value)))}
        min={50}
        max={300}
        step={10}
      />

      {/* Window Size */}
      <ParameterControl
        label="Context Window"
        description="Size of context window (words on each side)"
        type="number"
        value={parameters.window}
        onChange={(value) => onChange('window', parseInt(String(value)))}
        min={2}
        max={10}
        step={1}
      />

      {/* Min Count */}
      <ParameterControl
        label="Min Word Frequency"
        description="Ignore words with frequency below this"
        type="number"
        value={parameters.min_count}
        onChange={(value) => onChange('min_count', parseInt(String(value)))}
        min={1}
        max={20}
        step={1}
      />

      {/* Training Algorithm */}
      <ParameterControl
        label="Training Algorithm"
        description="CBOW or Skip-gram method"
        type="select"
        value={String(parameters.sg)}
        onChange={(value) => onChange('sg', parseInt(String(value)))}
        options={[
          { label: 'CBOW (slower, better for small datasets)', value: '0' },
          { label: 'Skip-gram (faster, better for large datasets)', value: '1' },
        ]}
      />

      {/* Epochs */}
      <ParameterControl
        label="Training Epochs"
        description="Number of training iterations"
        type="number"
        value={parameters.epochs}
        onChange={(value) => onChange('epochs', parseInt(String(value)))}
        min={5}
        max={50}
        step={1}
      />

      {/* Random State */}
      <ParameterControl
        label="Random Seed"
        description="For reproducible results"
        type="number"
        value={parameters.random_state}
        onChange={(value) => onChange('random_state', parseInt(String(value)))}
        min={0}
        max={9999}
        step={1}
      />

      {/* Custom Corpus */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Use Custom Corpus</Label>
          <Switch
            checked={parameters.use_custom_corpus}
            onCheckedChange={(checked) => onChange('use_custom_corpus', checked)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Train on custom text instead of default corpus
        </p>
      </div>

      {parameters.use_custom_corpus && (
        <div className="space-y-2">
          <Label>Custom Text Corpus</Label>
          <Textarea
            value={parameters.custom_corpus || ''}
            onChange={(e) => onChange('custom_corpus', e.target.value)}
            placeholder="Enter text corpus (sentences separated by newlines)..."
            rows={6}
            className="font-mono text-sm"
          />
          <p className="text-xs text-muted-foreground">
            Multiple sentences for better training results
          </p>
        </div>
      )}

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>CBOW:</strong> Predicts target word from context (slower, better for small data)
        </p>
        <p className="text-sm text-blue-900 dark:text-blue-100 mt-1">
          <strong>Skip-gram:</strong> Predicts context from target word (faster, better for large data)
        </p>
      </div>
    </div>
  );
}
