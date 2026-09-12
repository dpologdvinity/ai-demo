export interface Algorithm {
  id: string;
  name: string;
  description: string;
  category: AlgorithmCategory;
  complexity?: {
    time: string;
    space: string;
  };
  parameters?: AlgorithmParameter[];
}

export interface AlgorithmParameter {
  name: string;
  type: 'number' | 'string' | 'boolean' | 'array' | 'select';
  label: string;
  defaultValue?: any;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ value: string; label: string }>;
  required?: boolean;
}

export enum AlgorithmCategory {
  ML = 'ml',
  DeepLearning = 'deep-learning',
  NLP = 'nlp',
  ComputerVision = 'computer-vision',
  ReinforcementLearning = 'reinforcement-learning',
}

export interface AlgorithmExecution {
  algorithmId: string;
  parameters: Record<string, any>;
  timestamp: string;
}

export interface AlgorithmResult {
  success: boolean;
  data?: any;
  visualization?: VisualizationData;
  metrics?: Record<string, number>;
  error?: string;
  executionTime?: number;
}

export interface VisualizationData {
  type: 'line' | 'bar' | 'scatter' | 'heatmap' | 'network' | 'image';
  data: any;
  config?: Record<string, any>;
}

export interface CategoryInfo {
  id: AlgorithmCategory;
  name: string;
  description: string;
  icon: string;
  algorithms: Algorithm[];
}

export interface ApiResponse<T> {
  data: T;
  message?: string;
  status: number;
}

export interface ApiError {
  message: string;
  status: number;
  errors?: Record<string, string[]>;
}
