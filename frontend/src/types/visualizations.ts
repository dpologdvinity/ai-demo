/**
 * Common type definitions for data visualization components
 */

// Chart data types
export interface ChartDataPoint {
  [key: string]: number | string;
}

export type ChartData = ChartDataPoint[];

// Clustering types
export interface ClusterPoint {
  x: number;
  y: number;
  cluster?: number | string;
  label?: string;
}

export type ClusterData = ClusterPoint[];

// Neural network types
export interface NetworkLayer {
  name: string;
  nodes: number;
  activation?: string;
}

export type NetworkWeights = number[][][]; // [layer][from_node][to_node]

export interface NetworkArchitecture {
  layers: NetworkLayer[];
  weights?: NetworkWeights;
}

// Confusion matrix types
export type ConfusionMatrixData = number[][];

export interface ClassificationResult {
  predicted: string | number;
  actual: string | number;
  confidence?: number;
}

// Training metrics types
export interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy?: number;
  val_loss?: number;
  val_accuracy?: number;
  [key: string]: number | undefined;
}

export type TrainingHistory = TrainingMetrics[];

// Algorithm parameters types
export interface AlgorithmParameter {
  name: string;
  value: number | string;
  min?: number;
  max?: number;
  step?: number;
  type: "slider" | "number" | "select";
  options?: Array<{ value: string | number; label: string }>;
  description?: string;
}

export type AlgorithmParameters = Record<string, AlgorithmParameter>;

// Results types
export interface Metric {
  label: string;
  value: string | number;
  description?: string;
  format?: (value: number) => string;
}

export interface Prediction {
  label: string;
  value: string | number;
  confidence?: number;
  probability?: number;
}

export interface AlgorithmResults {
  metrics: Metric[];
  predictions?: Prediction[];
  confusionMatrix?: ConfusionMatrixData;
  trainingHistory?: TrainingHistory;
  metadata?: Record<string, any>;
}

// Time series types
export interface TimeSeriesPoint {
  timestamp: number | string;
  value: number;
  label?: string;
}

export type TimeSeriesData = TimeSeriesPoint[];

// Regression types
export interface RegressionPoint {
  x: number;
  y: number;
  predicted?: number;
  residual?: number;
}

export type RegressionData = RegressionPoint[];

// Feature importance types
export interface FeatureImportance {
  feature: string;
  importance: number;
  rank?: number;
}

export type FeatureImportanceData = FeatureImportance[];

// Dataset types
export interface Dataset {
  features: number[][];
  labels: (number | string)[];
  featureNames?: string[];
  labelNames?: string[];
}

export interface TrainTestSplit {
  train: Dataset;
  test: Dataset;
  validation?: Dataset;
}
