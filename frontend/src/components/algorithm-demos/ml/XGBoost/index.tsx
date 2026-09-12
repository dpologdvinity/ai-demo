import { useState } from 'react';
import { AlgorithmLayout } from '@/components/common/AlgorithmLayout';
import { Controls } from './Controls';
import { Visualization } from './Visualization';
import { Documentation } from './Documentation';
import { apiService } from '@/services/api';
import { useQuery } from '@tanstack/react-query';

interface XGBoostResult {
  success: boolean;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
  predictions: number[];
  visualization_data: {
    confusion_matrix: number[][];
    feature_importance: {
      features: string[];
      importance: number[];
    };
    learning_curves: {
      n_estimators: number[];
      train_accuracy: number[];
      test_accuracy: number[];
    };
    class_probabilities: number[][];
    target_names: string[];
  };
  execution_time_ms: number;
  parameters_used: Record<string, any>;
  error?: string;
}

export function XGBoost() {
  const [result, setResult] = useState<XGBoostResult | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [parameters, setParameters] = useState({
    n_estimators: 100,
    learning_rate: 0.1,
    max_depth: 6,
    subsample: 1.0,
    dataset_name: 'wine',
    normalize: true,
  });

  // Fetch algorithm info
  const { data: algorithmInfo } = useQuery({
    queryKey: ['algorithm-info', 'xgboost'],
    queryFn: () => apiService.getAlgorithmInfo('ml', 'xgboost'),
  });

  const handleTrain = async () => {
    setIsTraining(true);
    setError(null);

    try {
      const response = await apiService.trainAlgorithm('ml', 'xgboost', parameters);
      setResult(response as XGBoostResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed');
    } finally {
      setIsTraining(false);
    }
  };

  const handleParameterChange = (name: string, value: number | string | boolean) => {
    setParameters((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  return (
    <AlgorithmLayout
      title="Gradient Boosting (XGBoost)"
      description="Ensemble method that builds trees sequentially to correct errors"
      sections={{
        parameters: (
          <Controls
            parameters={parameters}
            onParameterChange={handleParameterChange}
            onTrain={handleTrain}
            isTraining={isTraining}
          />
        ),
        visualization: (
          <Visualization
            result={result}
            isTraining={isTraining}
            error={error}
          />
        ),
      }}
    >
      <Documentation metadata={algorithmInfo?.metadata} />
    </AlgorithmLayout>
  );
}

export default XGBoost;
