import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';

interface GloVeParams {
  embedding_dim: number;
  top_k: number;
  query_word: string;
  analogy_word_a: string;
  analogy_word_b: string;
  analogy_word_c: string;
}

interface ControlsProps {
  parameters: GloVeParams;
  onChange: (name: keyof GloVeParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Embedding Dimension */}
      <ParameterControl
        label="Embedding Dimension"
        description="Vector dimensionality"
        type="select"
        value={String(parameters.embedding_dim)}
        onChange={(value) => onChange('embedding_dim', parseInt(String(value)))}
        options={[
          { label: '50D', value: '50' },
          { label: '100D', value: '100' },
          { label: '200D', value: '200' },
          { label: '300D', value: '300' },
        ]}
      />

      {/* Top K */}
      <ParameterControl
        label="Similar Words to Show"
        description="Number of similar words to display"
        type="number"
        value={parameters.top_k}
        onChange={(value) => onChange('top_k', parseInt(String(value)))}
        min={5}
        max={20}
        step={1}
      />

      {/* Query Word */}
      <div className="space-y-2">
        <Label>Query Word</Label>
        <Input
          value={parameters.query_word}
          onChange={(e) => onChange('query_word', e.target.value.toLowerCase())}
          placeholder="Enter word (e.g., king)"
          className="font-mono text-sm"
        />
        <p className="text-xs text-muted-foreground">
          Find similar words for this word
        </p>
      </div>

      {/* Analogy Section */}
      <div className="border-t pt-4">
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Word Analogy
        </h4>
        <p className="text-xs text-gray-600 dark:text-gray-400 mb-3">
          Find: {parameters.analogy_word_a} - {parameters.analogy_word_b} + {parameters.analogy_word_c}
        </p>

        {/* Analogy Word A */}
        <div className="space-y-2 mb-3">
          <Label className="text-xs">Word A</Label>
          <Input
            value={parameters.analogy_word_a}
            onChange={(e) => onChange('analogy_word_a', e.target.value.toLowerCase())}
            placeholder="king"
            className="font-mono text-sm"
          />
        </div>

        {/* Analogy Word B */}
        <div className="space-y-2 mb-3">
          <Label className="text-xs">Word B (subtract)</Label>
          <Input
            value={parameters.analogy_word_b}
            onChange={(e) => onChange('analogy_word_b', e.target.value.toLowerCase())}
            placeholder="man"
            className="font-mono text-sm"
          />
        </div>

        {/* Analogy Word C */}
        <div className="space-y-2">
          <Label className="text-xs">Word C (add)</Label>
          <Input
            value={parameters.analogy_word_c}
            onChange={(e) => onChange('analogy_word_c', e.target.value.toLowerCase())}
            placeholder="woman"
            className="font-mono text-sm"
          />
        </div>
      </div>

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Example analogy:</strong> king - man + woman = queen
        </p>
        <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
          GloVe learns semantic relationships between words
        </p>
      </div>
    </div>
  );
}
