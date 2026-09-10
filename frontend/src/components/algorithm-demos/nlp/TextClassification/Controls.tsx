import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';

interface TextClassificationParams {
  classifier_type: string;
  max_features: number;
  test_size: number;
  ngram_range: [number, number];
  use_custom_dataset: boolean;
  custom_texts?: string[];
  custom_labels?: string[];
}

interface ControlsProps {
  parameters: TextClassificationParams;
  onChange: (name: keyof TextClassificationParams, value: any) => void;
  algorithmInfo?: any;
}

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  return (
    <div className="space-y-4">
      {/* Classifier Type */}
      <ParameterControl
        label="Classifier Type"
        description="Machine learning classifier to use"
        type="select"
        value={parameters.classifier_type}
        onChange={(value) => onChange('classifier_type', value)}
        options={[
          { label: 'Naive Bayes', value: 'naive_bayes' },
          { label: 'Logistic Regression', value: 'logistic_regression' },
          { label: 'SVM', value: 'svm' },
        ]}
      />

      {/* Max Features */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Max TF-IDF Features</Label>
          <span className="text-sm text-muted-foreground">
            {parameters.max_features}
          </span>
        </div>
        <Slider
          value={[parameters.max_features]}
          onValueChange={([value]) => onChange('max_features', value)}
          min={100}
          max={5000}
          step={100}
        />
        <p className="text-xs text-muted-foreground">
          Maximum number of features for vectorization
        </p>
      </div>

      {/* Test Size */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Test Set Size</Label>
          <span className="text-sm text-muted-foreground">
            {(parameters.test_size * 100).toFixed(0)}%
          </span>
        </div>
        <Slider
          value={[parameters.test_size]}
          onValueChange={([value]) => onChange('test_size', value)}
          min={0.1}
          max={0.4}
          step={0.05}
        />
        <p className="text-xs text-muted-foreground">
          Ratio of data used for testing
        </p>
      </div>

      {/* N-gram Range */}
      <ParameterControl
        label="N-gram Range"
        description="Tokenization n-gram range"
        type="select"
        value={`${parameters.ngram_range[0]}-${parameters.ngram_range[1]}`}
        onChange={(value: any) => {
          const [min, max] = String(value).split('-').map(Number) as [number, number];
          onChange('ngram_range', [min, max]);
        }}
        options={[
          { label: 'Unigrams (1-1)', value: '1-1' },
          { label: 'Unigrams + Bigrams (1-2)', value: '1-2' },
          { label: 'Unigrams + Bigrams + Trigrams (1-3)', value: '1-3' },
        ]}
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
          Train with custom texts and labels
        </p>
      </div>

      {parameters.use_custom_dataset && (
        <div className="pt-2 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <p className="text-xs text-blue-900 dark:text-blue-100">
            <strong>Note:</strong> Custom dataset feature requires preparing data outside this interface.
          </p>
        </div>
      )}

      {!parameters.use_custom_dataset && (
        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <p className="text-xs text-blue-900 dark:text-blue-100">
            <strong>Dataset:</strong> Movie reviews with positive/negative labels (2000 samples)
          </p>
        </div>
      )}
    </div>
  );
}
