import { ParameterControl } from '@/components/common/ParameterControl';

interface AStarControlsProps {
  parameters: {
    grid_size: number;
    obstacle_density: number;
    heuristic: 'manhattan' | 'euclidean' | 'chebyshev';
    random_state: number;
  };
  onParameterChange: (name: string, value: number | string | boolean) => void;
  onTrain: () => void;
  isTraining: boolean;
}

export default function Controls({
  parameters,
  onParameterChange,
  onTrain,
  isTraining,
}: AStarControlsProps) {
  return (
    <div className="space-y-4">
      <ParameterControl
        label="Grid Size"
        type="range"
        min={5}
        max={30}
        step={1}
        value={parameters.grid_size}
        onChange={(val) => onParameterChange('grid_size', val)}
        description="5 to 30"
      />

      <ParameterControl
        label="Obstacle Density"
        type="range"
        min={0}
        max={0.6}
        step={0.05}
        value={parameters.obstacle_density}
        onChange={(val) => onParameterChange('obstacle_density', val)}
        description="0 (clear) to 0.6 (50% obstacles)"
      />

      <ParameterControl
        label="Heuristic"
        type="select"
        value={parameters.heuristic}
        onChange={(val) => onParameterChange('heuristic', val)}
        options={[
          { value: 'manhattan', label: 'Manhattan' },
          { value: 'euclidean', label: 'Euclidean' },
          { value: 'chebyshev', label: 'Chebyshev' },
        ]}
      />

      <ParameterControl
        label="Random Seed"
        type="number"
        value={parameters.random_state}
        onChange={(val) => onParameterChange('random_state', val)}
        description="For reproducibility"
      />

      <button
        onClick={onTrain}
        disabled={isTraining}
        className="w-full px-4 py-2 bg-neon-cyan text-black font-semibold rounded-md hover:shadow-glow-cyan disabled:opacity-50 transition-all"
      >
        {isTraining ? 'Finding Path...' : 'Find Path'}
      </button>
    </div>
  );
}
