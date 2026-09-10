import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';

interface TokenizationParams {
  tokenizer_type: string;
  lowercase: boolean;
  remove_punctuation: boolean;
  remove_stopwords: boolean;
  max_tokens: number;
  text_index: number;
  compare_mode: boolean;
  custom_text?: string;
}

interface ControlsProps {
  parameters: TokenizationParams;
  onChange: (name: keyof TokenizationParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Tokenizer Type */}
      <ParameterControl
        label="Tokenizer Type"
        description="Select tokenization strategy"
        type="select"
        value={parameters.tokenizer_type}
        onChange={(value) => onChange('tokenizer_type', value)}
        options={[
          { label: 'Word', value: 'word' },
          { label: 'Whitespace', value: 'whitespace' },
          { label: 'Sentence', value: 'sentence' },
          { label: 'WordPiece', value: 'wordpiece' },
          { label: 'BPE', value: 'bpe' },
          { label: 'Character', value: 'character' },
          { label: 'spaCy', value: 'spacy' },
        ]}
      />

      {/* Lowercase */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Lowercase</Label>
          <Switch
            checked={parameters.lowercase}
            onCheckedChange={(checked) => onChange('lowercase', checked)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Convert text to lowercase before tokenization
        </p>
      </div>

      {/* Remove Punctuation */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Remove Punctuation</Label>
          <Switch
            checked={parameters.remove_punctuation}
            onCheckedChange={(checked) => onChange('remove_punctuation', checked)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Strip punctuation from tokens
        </p>
      </div>

      {/* Remove Stopwords */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Remove Stopwords</Label>
          <Switch
            checked={parameters.remove_stopwords}
            onCheckedChange={(checked) => onChange('remove_stopwords', checked)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Filter common stopwords (the, a, is, etc.)
        </p>
      </div>

      {/* Max Tokens */}
      <ParameterControl
        label="Max Tokens to Display"
        description="Maximum number of tokens to show"
        type="number"
        value={parameters.max_tokens}
        onChange={(value) => onChange('max_tokens', parseInt(String(value)))}
        min={10}
        max={500}
        step={10}
      />

      {/* Compare Mode */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Compare Tokenizers</Label>
          <Switch
            checked={parameters.compare_mode}
            onCheckedChange={(checked) => onChange('compare_mode', checked)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Show results from all tokenizer types
        </p>
      </div>

      {/* Custom Text */}
      {!parameters.compare_mode && (
        <div className="space-y-2">
          <Label>Custom Text (Optional)</Label>
          <Textarea
            value={parameters.custom_text || ''}
            onChange={(e) => onChange('custom_text', e.target.value)}
            placeholder="Enter custom text to tokenize..."
            rows={4}
            className="font-mono text-sm"
          />
          <p className="text-xs text-muted-foreground">
            Leave empty to use sample text
          </p>
        </div>
      )}
    </div>
  );
}
