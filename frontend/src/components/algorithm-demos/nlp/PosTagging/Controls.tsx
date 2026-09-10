import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';

interface POSTaggingParams {
  tagger: string;
  text_index: number;
  show_fine_grained: boolean;
  show_dependencies: boolean;
  tag_scheme: string;
  use_custom_text: boolean;
  custom_text?: string;
}

interface ControlsProps {
  parameters: POSTaggingParams;
  onChange: (name: keyof POSTaggingParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Tagger Type */}
      <ParameterControl
        label="Tagger Type"
        description="POS tagging library to use"
        type="select"
        value={parameters.tagger}
        onChange={(value) => onChange('tagger', value)}
        options={[
          { label: 'spaCy', value: 'spacy' },
          { label: 'NLTK', value: 'nltk' },
          { label: 'Universal', value: 'universal' },
        ]}
      />

      {/* Tag Scheme */}
      <ParameterControl
        label="Tag Scheme"
        description="POS tag scheme to use"
        type="select"
        value={parameters.tag_scheme}
        onChange={(value) => onChange('tag_scheme', value)}
        options={[
          { label: 'Penn Treebank', value: 'penn' },
          { label: 'Universal', value: 'universal' },
        ]}
      />

      {/* Text Index (Sample) */}
      <ParameterControl
        label="Sample Text Index"
        description="Select from built-in samples (0-24)"
        type="number"
        value={parameters.text_index}
        onChange={(value) => onChange('text_index', parseInt(String(value)))}
        min={0}
        max={24}
        step={1}
      />

      {/* Show Fine-Grained */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Fine-Grained Tags</Label>
          <Switch
            checked={parameters.show_fine_grained}
            onCheckedChange={(checked) => onChange('show_fine_grained', checked)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Show detailed Penn Treebank tags
        </p>
      </div>

      {/* Show Dependencies */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Dependency Parsing</Label>
          <Switch
            checked={parameters.show_dependencies}
            onCheckedChange={(checked) => onChange('show_dependencies', checked)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Include dependency parsing information
        </p>
      </div>

      {/* Use Custom Text */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Use Custom Text</Label>
          <Switch
            checked={parameters.use_custom_text}
            onCheckedChange={(checked) => onChange('use_custom_text', checked)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Analyze your own text instead of samples
        </p>
      </div>

      {/* Custom Text Input */}
      {parameters.use_custom_text && (
        <div className="space-y-2">
          <Label>Custom Text</Label>
          <Textarea
            value={parameters.custom_text || ''}
            onChange={(e) => onChange('custom_text', e.target.value)}
            placeholder="Enter text to tag..."
            rows={6}
            className="font-mono text-sm"
          />
        </div>
      )}
    </div>
  );
}
