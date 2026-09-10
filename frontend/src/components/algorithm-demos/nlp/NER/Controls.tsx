import React from 'react';
import { ParameterControl } from '@/components/common/ParameterControl';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';

interface NERParams {
  model_name: string;
  entity_types: string[];
  confidence_threshold: number;
  text_index: number;
  merge_entities: boolean;
  use_custom_text: boolean;
  custom_text?: string;
}

interface ControlsProps {
  parameters: NERParams;
  onChange: (name: keyof NERParams, value: any) => void;
  algorithmInfo?: any;
}

const ENTITY_TYPES = [
  'PERSON',
  'ORG',
  'GPE',
  'DATE',
  'TIME',
  'MONEY',
  'PERCENT',
  'FACILITY',
  'PRODUCT',
  'EVENT',
  'LAW',
  'LANGUAGE',
  'NORP',
  'CARDINAL',
  'ORDINAL',
  'QUANTITY',
];

export function Controls({ parameters, onChange, algorithmInfo }: ControlsProps) {
  const handleEntityTypeToggle = (type: string) => {
    const updated = parameters.entity_types.includes(type)
      ? parameters.entity_types.filter(t => t !== type)
      : [...parameters.entity_types, type];
    onChange('entity_types', updated);
  };

  return (
    <div className="space-y-4">
      {/* Model Selection */}
      <ParameterControl
        label="spaCy Model"
        description="Language model for NER"
        type="select"
        value={parameters.model_name}
        onChange={(value) => onChange('model_name', value)}
        options={[
          { label: 'en_core_web_sm (Small)', value: 'en_core_web_sm' },
          { label: 'en_core_web_md (Medium)', value: 'en_core_web_md' },
        ]}
      />

      {/* Confidence Threshold */}
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
        />
        <p className="text-xs text-muted-foreground">
          Minimum confidence score for entities
        </p>
      </div>

      {/* Text Index */}
      <ParameterControl
        label="Sample Text Index"
        description="Select from built-in samples (0-29)"
        type="number"
        value={parameters.text_index}
        onChange={(value) => onChange('text_index', parseInt(String(value)))}
        min={0}
        max={29}
        step={1}
      />

      {/* Merge Entities */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Merge Adjacent Entities</Label>
          <Switch
            checked={parameters.merge_entities}
            onCheckedChange={(checked) => onChange('merge_entities', checked)}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Combine adjacent entities of the same type
        </p>
      </div>

      {/* Entity Types Selection */}
      <div className="space-y-2">
        <Label>Entity Types to Extract</Label>
        <div className="grid grid-cols-2 gap-2">
          {ENTITY_TYPES.map((type) => (
            <button
              key={type}
              onClick={() => handleEntityTypeToggle(type)}
              className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                parameters.entity_types.includes(type)
                  ? 'bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-100'
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
              }`}
            >
              {type}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          {parameters.entity_types.length} types selected
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
            placeholder="Enter text to extract entities from..."
            rows={6}
            className="font-mono text-sm"
          />
        </div>
      )}
    </div>
  );
}
