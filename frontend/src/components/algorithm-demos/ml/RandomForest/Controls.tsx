import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { RandomForestParams } from './index';

interface ControlsProps {
  onTrain: (params: RandomForestParams) => void;
  isTraining: boolean;
}

export const Controls: React.FC<ControlsProps> = ({ onTrain, isTraining }) => {
  const [params, setParams] = useState<RandomForestParams>({
    n_estimators: 100,
    max_depth: null,
    min_samples_split: 2,
    max_features: 'sqrt',
    random_state: 42,
  });

  const [maxDepthEnabled, setMaxDepthEnabled] = useState(false);

  const handleTrain = () => {
    const trainingParams = {
      ...params,
      max_depth: maxDepthEnabled ? params.max_depth : null,
    };
    onTrain(trainingParams);
  };

  return (
    <div className="space-y-6">
      {/* Number of Estimators */}
      <div className="space-y-2">
        <Label htmlFor="n_estimators">
          Number of Trees: {params.n_estimators}
        </Label>
        <Slider
          id="n_estimators"
          min={10}
          max={500}
          step={10}
          value={[params.n_estimators]}
          onValueChange={([value]) =>
            setParams({ ...params, n_estimators: value })
          }
          disabled={isTraining}
        />
        <p className="text-xs text-muted-foreground">
          More trees generally improve accuracy but increase training time
        </p>
      </div>

      {/* Max Depth */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="max_depth">Maximum Depth</Label>
          <label className="flex items-center space-x-2 text-sm">
            <input
              type="checkbox"
              checked={maxDepthEnabled}
              onChange={(e) => {
                setMaxDepthEnabled(e.target.checked);
                if (!e.target.checked) {
                  setParams({ ...params, max_depth: null });
                } else {
                  setParams({ ...params, max_depth: 10 });
                }
              }}
              disabled={isTraining}
              className="rounded"
            />
            <span>Limit depth</span>
          </label>
        </div>
        {maxDepthEnabled && (
          <>
            <Input
              id="max_depth"
              type="number"
              min={1}
              max={30}
              value={params.max_depth || 10}
              onChange={(e) =>
                setParams({ ...params, max_depth: parseInt(e.target.value) })
              }
              disabled={isTraining}
            />
            <p className="text-xs text-muted-foreground">
              Limiting depth helps prevent overfitting
            </p>
          </>
        )}
        {!maxDepthEnabled && (
          <p className="text-xs text-muted-foreground">
            Unlimited depth (None) - trees grow until pure
          </p>
        )}
      </div>

      {/* Min Samples Split */}
      <div className="space-y-2">
        <Label htmlFor="min_samples_split">
          Min Samples Split: {params.min_samples_split}
        </Label>
        <Slider
          id="min_samples_split"
          min={2}
          max={20}
          step={1}
          value={[params.min_samples_split]}
          onValueChange={([value]) =>
            setParams({ ...params, min_samples_split: value })
          }
          disabled={isTraining}
        />
        <p className="text-xs text-muted-foreground">
          Minimum samples required to split an internal node
        </p>
      </div>

      {/* Max Features */}
      <div className="space-y-2">
        <Label htmlFor="max_features">Max Features</Label>
        <Select
          value={params.max_features}
          onValueChange={(value) =>
            setParams({ ...params, max_features: value })
          }
          disabled={isTraining}
        >
          <SelectTrigger id="max_features">
            <SelectValue placeholder="Select max features" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="sqrt">Square Root (√n)</SelectItem>
            <SelectItem value="log2">Log2 (log₂n)</SelectItem>
            <SelectItem value="None">All Features</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Number of features to consider for the best split
        </p>
      </div>

      {/* Random State */}
      <div className="space-y-2">
        <Label htmlFor="random_state">Random Seed</Label>
        <Input
          id="random_state"
          type="number"
          value={params.random_state}
          onChange={(e) =>
            setParams({ ...params, random_state: parseInt(e.target.value) })
          }
          disabled={isTraining}
        />
        <p className="text-xs text-muted-foreground">
          For reproducible results
        </p>
      </div>

      {/* Train Button */}
      <Button
        onClick={handleTrain}
        disabled={isTraining}
        className="w-full"
        size="lg"
      >
        {isTraining ? 'Training...' : 'Train Model'}
      </Button>
    </div>
  );
};
