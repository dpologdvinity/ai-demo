import { ParameterControl } from '@/components/common/ParameterControl';

interface MinimaxControlsProps {
  parameters: {
    opponent: 'random' | 'optimal';
    ai_starts: boolean;
    use_alpha_beta: boolean;
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
}: MinimaxControlsProps) {
  return (
    <div className="space-y-4">
      <ParameterControl
        label="Opponent Strategy"
        type="select"
        value={parameters.opponent}
        onChange={(val) => onParameterChange('opponent', val)}
        options={[
          { value: 'random', label: 'Random' },
          { value: 'optimal', label: 'Optimal' },
        ]}
      />

      <ParameterControl
        label="AI Starts First"
        type="boolean"
        value={parameters.ai_starts}
        onChange={(val) => onParameterChange('ai_starts', val)}
      />

      <ParameterControl
        label="Use Alpha-Beta Pruning"
        type="boolean"
        value={parameters.use_alpha_beta}
        onChange={(val) => onParameterChange('use_alpha_beta', val)}
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
        className="w-full px-4 py-2 bg-neon-magenta text-black font-semibold rounded-md hover:shadow-glow-magenta disabled:opacity-50 transition-all"
      >
        {isTraining ? 'Playing Game...' : 'Play Game'}
      </button>
    </div>
  );
}
