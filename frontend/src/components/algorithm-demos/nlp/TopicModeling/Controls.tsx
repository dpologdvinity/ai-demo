import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';

interface TopicModelingParams {
  n_topics: number;
  max_iterations: number;
  alpha: string | number;
  beta: string | number;
  min_df: number;
  max_df: number;
  use_custom_documents: boolean;
  custom_documents?: string[];
}

interface ControlsProps {
  parameters: TopicModelingParams;
  onChange: (name: keyof TopicModelingParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Number of Topics */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Number of Topics</Label>
          <span className="text-sm text-muted-foreground">
            {parameters.n_topics}
          </span>
        </div>
        <Slider
          value={[parameters.n_topics]}
          onValueChange={([value]) => onChange('n_topics', value)}
          min={2}
          max={20}
          step={1}
        />
        <p className="text-xs text-muted-foreground">
          Number of topics to discover (2-20)
        </p>
      </div>

      {/* Max Iterations */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Max Iterations</Label>
          <span className="text-sm text-muted-foreground">
            {parameters.max_iterations}
          </span>
        </div>
        <Slider
          value={[parameters.max_iterations]}
          onValueChange={([value]) => onChange('max_iterations', value)}
          min={20}
          max={500}
          step={20}
        />
        <p className="text-xs text-muted-foreground">
          Maximum LDA iterations (20-500)
        </p>
      </div>

      {/* Alpha */}
      <ParameterControl
        label="Alpha (Doc-Topic)"
        description="Document-topic density"
        type="select"
        value={String(parameters.alpha)}
        onChange={(value: any) => onChange('alpha', value === 'auto' ? 'auto' : parseFloat(value))}
        options={[
          { label: 'Auto', value: 'auto' },
          { label: '0.01', value: '0.01' },
          { label: '0.05', value: '0.05' },
          { label: '0.1', value: '0.1' },
        ]}
      />

      {/* Beta */}
      <ParameterControl
        label="Beta (Topic-Word)"
        description="Topic-word density"
        type="select"
        value={String(parameters.beta)}
        onChange={(value: any) => onChange('beta', value === 'auto' ? 'auto' : parseFloat(value))}
        options={[
          { label: 'Auto', value: 'auto' },
          { label: '0.01', value: '0.01' },
          { label: '0.05', value: '0.05' },
          { label: '0.1', value: '0.1' },
        ]}
      />

      {/* Min Document Frequency */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Min Document Frequency</Label>
          <span className="text-sm text-muted-foreground">
            {parameters.min_df}
          </span>
        </div>
        <Slider
          value={[parameters.min_df]}
          onValueChange={([value]) => onChange('min_df', value)}
          min={1}
          max={10}
          step={1}
        />
        <p className="text-xs text-muted-foreground">
          Minimum documents a word must appear in
        </p>
      </div>

      {/* Max Document Frequency */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Max Document Frequency</Label>
          <span className="text-sm text-muted-foreground">
            {(parameters.max_df * 100).toFixed(0)}%
          </span>
        </div>
        <Slider
          value={[parameters.max_df]}
          onValueChange={([value]) => onChange('max_df', value)}
          min={0.5}
          max={1.0}
          step={0.05}
        />
        <p className="text-xs text-muted-foreground">
          Maximum proportion of documents containing a word
        </p>
      </div>

      {/* Use Custom Documents */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Use Custom Documents</Label>
          <Switch
            checked={parameters.use_custom_documents}
            onCheckedChange={(checked) => onChange('use_custom_documents', checked)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Analyze custom document collection
        </p>
      </div>

      {!parameters.use_custom_documents && (
        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <p className="text-xs text-blue-900 dark:text-blue-100">
            <strong>Dataset:</strong> BBC News articles (2225 documents) across 5 categories
          </p>
        </div>
      )}
    </div>
  );
}
