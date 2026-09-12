import React, { useState } from 'react';
import { AlgorithmLayout } from '@/components/common/AlgorithmLayout';
import { apiService } from '@/services/api';
import { Controls } from './Controls';
import { Visualization } from './Visualization';
import { Documentation } from './Documentation';

export interface RandomForestParams {
  n_estimators: number;
  max_depth: number | null;
  min_samples_split: number;
  max_features: string;
  random_state: number;
}

export interface RandomForestResults {
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
  predictions: number[];
  feature_importance: Record<string, number>;
  confusion_matrix: number[][];
  execution_time_ms: number;
  model_info: {
    n_estimators: number;
    max_depth: number | string;
    min_samples_split: number;
    max_features: string;
    total_trees: number;
    n_features: number;
    n_classes: number;
    class_names: string[];
  };
}

export const RandomForest: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [results, setResults] = useState<RandomForestResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTrain = async (params: RandomForestParams) => {
    setIsTraining(true);
    setError(null);

    try {
      const data = await apiService.trainAlgorithm('ml', 'random-forest', params);
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      console.error('Training error:', err);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <AlgorithmLayout
      title="Random Forest"
      description="Ensemble learning method using multiple decision trees for robust predictions"
      sections={{
        parameters: <Controls onTrain={handleTrain} isTraining={isTraining} />,
        visualization: (
          <Visualization
            results={results}
            isTraining={isTraining}
            error={error}
          />
        ),
      }}
    >
      <Documentation />
    </AlgorithmLayout>
  );
};

export default RandomForest;
