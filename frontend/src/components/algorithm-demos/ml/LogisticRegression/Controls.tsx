import React from 'react';
import { Play } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/common/Card';
import Button from '@/components/common/Button';

interface ControlsProps {
  parameters: {
    C: number;
    penalty: string;
    max_iter: number;
    solver: string;
  };
  onParameterChange: (name: string, value: any) => void;
  onTrain: () => void;
  isTraining: boolean;
  datasetName: string;
  onDatasetChange: (dataset: string) => void;
  availableDatasets: string[];
}

export function Controls({
  parameters,
  onParameterChange,
  onTrain,
  isTraining,
  datasetName,
  onDatasetChange,
  availableDatasets,
}: ControlsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Configuration</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Dataset Selection */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Dataset
            </label>
            <select
              value={datasetName}
              onChange={(e) => onDatasetChange(e.target.value)}
              className="w-full px-3 py-2 border border-border rounded-md bg-background"
              disabled={isTraining}
            >
              {availableDatasets.map((dataset) => (
                <option key={dataset} value={dataset}>
                  {dataset.charAt(0).toUpperCase() + dataset.slice(1)}
                </option>
              ))}
            </select>
          </div>

          {/* Regularization Strength (C) */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Regularization Strength (C): {parameters.C.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.01"
              max="10"
              step="0.1"
              value={parameters.C}
              onChange={(e) =>
                onParameterChange('C', parseFloat(e.target.value))
              }
              className="w-full"
              disabled={isTraining}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Smaller values = stronger regularization
            </p>
          </div>

          {/* Penalty Type */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Penalty Type
            </label>
            <select
              value={parameters.penalty}
              onChange={(e) => onParameterChange('penalty', e.target.value)}
              className="w-full px-3 py-2 border border-border rounded-md bg-background"
              disabled={isTraining}
            >
              <option value="l2">L2 (Ridge)</option>
              <option value="l1">L1 (Lasso)</option>
            </select>
            <p className="text-xs text-muted-foreground mt-1">
              L1 promotes sparsity, L2 promotes smaller weights
            </p>
          </div>

          {/* Maximum Iterations */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Maximum Iterations: {parameters.max_iter}
            </label>
            <input
              type="range"
              min="50"
              max="500"
              step="10"
              value={parameters.max_iter}
              onChange={(e) =>
                onParameterChange('max_iter', parseInt(e.target.value))
              }
              className="w-full"
              disabled={isTraining}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Maximum iterations for solver convergence
            </p>
          </div>

          {/* Solver Algorithm */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Solver Algorithm
            </label>
            <select
              value={parameters.solver}
              onChange={(e) => onParameterChange('solver', e.target.value)}
              className="w-full px-3 py-2 border border-border rounded-md bg-background"
              disabled={isTraining}
            >
              <option value="lbfgs">LBFGS (L2 only)</option>
              <option value="liblinear">Liblinear (L1/L2, small datasets)</option>
              <option value="saga">SAGA (L1/L2, large datasets)</option>
            </select>
            <p className="text-xs text-muted-foreground mt-1">
              Choose optimizer based on dataset size and penalty
            </p>
          </div>

          {/* Train Button */}
          <Button
            onClick={onTrain}
            disabled={isTraining}
            className="w-full"
          >
            {isTraining ? (
              <>
                <span className="animate-spin mr-2">⏳</span>
                Training...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Train Model
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
