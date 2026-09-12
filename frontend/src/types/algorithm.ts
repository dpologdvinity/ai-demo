export type AlgorithmCategory =
  | 'ml'
  | 'deep_learning'
  | 'nlp'
  | 'computer_vision'
  | 'reinforcement_learning';

export type DifficultyLevel = 'Beginner' | 'Intermediate' | 'Advanced';

export interface Algorithm {
  id: string;
  name: string;
  slug: string;
  category: AlgorithmCategory;
  description: string;
  difficulty: DifficultyLevel;
  tags: string[];
  useCases: string[];
  complexity: {
    time: string;
    space: string;
  };
}

export interface AlgorithmParameter {
  name: string;
  label: string;
  type: 'number' | 'select' | 'boolean' | 'range';
  default: any;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ label: string; value: any }>;
  description: string;
}

export interface AlgorithmResult {
  metrics: Record<string, number>;
  predictions?: any[];
  visualizationData: any;
  executionTime: number;
  parameters: Record<string, any>;
}

export interface TrainingProgress {
  epoch?: number;
  iteration?: number;
  loss?: number;
  accuracy?: number;
  timestamp: number;
  [key: string]: any;
}
