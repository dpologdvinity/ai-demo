import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';

interface TFIDFParams {
  max_features: number;
  ngram_range: [number, number];
  min_df: number;
  max_df: number;
  use_idf: boolean;
  normalize: boolean;
  custom_documents?: string[];
}

interface ControlsProps {
  parameters: TFIDFParams;
  onChange: (name: keyof TFIDFParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Max Features */}
      <ParameterControl
        label="Max Features"
        description="Maximum number of features to extract"
        type="number"
        value={parameters.max_features}
        onChange={(value) => onChange('max_features', parseInt(String(value)))}
        min={10}
        max={500}
        step={10}
      />

      {/* N-gram Range */}
      <ParameterControl
        label="N-gram Range"
        description="Use unigrams, bigrams, or trigrams"
        type="select"
        value={`${parameters.ngram_range[0]},${parameters.ngram_range[1]}`}
        onChange={(value) => {
          const [min, max] = String(value).split(',').map(Number);
          onChange('ngram_range', [min, max]);
        }}
        options={[
          { label: 'Unigrams (1,1)', value: '1,1' },
          { label: 'Unigrams + Bigrams (1,2)', value: '1,2' },
          { label: 'Unigrams + Bigrams + Trigrams (1,3)', value: '1,3' },
        ]}
      />

      {/* Min Document Frequency */}
      <ParameterControl
        label="Min Document Frequency"
        description="Minimum documents a term must appear in"
        type="number"
        value={parameters.min_df}
        onChange={(value) => onChange('min_df', parseInt(String(value)))}
        min={1}
        max={10}
        step={1}
      />

      {/* Max Document Frequency */}
      <ParameterControl
        label="Max Document Frequency"
        description="Maximum proportion of documents a term can appear in"
        type="range"
        value={parameters.max_df}
        onChange={(value) => onChange('max_df', parseFloat(String(value)))}
        min={0.5}
        max={1.0}
        step={0.1}
      />

      {/* Use IDF */}
      <ParameterControl
        label="Use IDF Weighting"
        description="Apply inverse document frequency weighting"
        type="select"
        value={parameters.use_idf ? 'true' : 'false'}
        onChange={(value) => onChange('use_idf', value === 'true')}
        options={[
          { label: 'Yes', value: 'true' },
          { label: 'No (TF only)', value: 'false' },
        ]}
      />

      {/* Normalize */}
      <ParameterControl
        label="Normalize Vectors"
        description="Normalize vectors to unit length"
        type="select"
        value={parameters.normalize ? 'true' : 'false'}
        onChange={(value) => onChange('normalize', value === 'true')}
        options={[
          { label: 'Yes', value: 'true' },
          { label: 'No', value: 'false' },
        ]}
      />

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Dataset:</strong> Sample documents provided by default
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          TF-IDF weights terms based on their frequency in documents and across the corpus
        </p>
      </div>
    </div>
  );
}
