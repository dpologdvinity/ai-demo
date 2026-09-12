import * as React from "react";
import { useMutation } from "@tanstack/react-query";
import {
  AlgorithmLayout,
  ParameterControl,
  CodeDisplay,
  ResultsPanel,
  LoadingSpinner,
  ErrorDisplay,
} from "@/components/common";
import {
  ConfusionMatrix,
  ScatterPlot,
} from "@/components/visualizations";
import { apiService } from "@/services/api";
import { AlgorithmCategory } from "@/types";

/**
 * Naive Bayes Algorithm Demonstration Component
 *
 * Demonstrates Gaussian Naive Bayes classifier with:
 * - Configurable variance smoothing parameter
 * - Multiple dataset options (Iris, Wine, Digits)
 * - Real-time training and visualization
 * - Probability distribution analysis
 * - Confusion matrix visualization
 */
export function NaiveBayes() {
  // Parameter state
  const [varSmoothing, setVarSmoothing] = React.useState(1e-9);
  const [dataset, setDataset] = React.useState("iris");
  const [normalize, setNormalize] = React.useState(true);

  // Results state
  const [results, setResults] = React.useState<any>(null);

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: async () => {
      const response = await apiService.trainAlgorithm(
        AlgorithmCategory.ML,
        "naive-bayes",
        {
          var_smoothing: varSmoothing,
          dataset_name: dataset,
          normalize: normalize,
          priors: null,
        }
      );
      return response;
    },
    onSuccess: (data) => {
      setResults(data);
    },
  });

  // Handle training
  const handleTrain = () => {
    trainMutation.mutate();
  };

  // Generate code example
  const generateCode = () => {
    return `
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_${dataset === 'iris' ? 'iris' : dataset === 'wine' ? 'wine' : 'digits'}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_${dataset === 'iris' ? 'iris' : dataset === 'wine' ? 'wine' : 'digits'}()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

${normalize ? `# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
` : ''}
# Create and train Naive Bayes classifier
model = GaussianNB(var_smoothing=${varSmoothing})
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))
    `.trim();
  };

  return (
    <AlgorithmLayout
      title="Naive Bayes Classifier"
      description="Probabilistic classifier based on Bayes' theorem with feature independence assumption"
      sections={{
        parameters: (
          <div className="space-y-4">
            <ParameterControl
              label="Variance Smoothing"
              value={varSmoothing}
              onChange={(val) => setVarSmoothing(Number(val))}
              min={1e-10}
              max={1e-8}
              step={1e-10}
              type="slider"
              description="Portion of largest variance added for stability"
            />

            <ParameterControl
              label="Dataset"
              value={dataset}
              onChange={setDataset}
              type="select"
              options={[
                { value: "iris", label: "Iris (3 classes, 4 features)" },
                { value: "wine", label: "Wine (3 classes, 13 features)" },
                { value: "digits", label: "Digits (10 classes, 64 features)" },
              ]}
              description="Dataset to use for training"
            />

            <ParameterControl
              label="Normalize Features"
              value={normalize}
              onChange={(val) => setNormalize(val === "true" || val === true)}
              type="select"
              options={[
                { value: "true", label: "Yes" },
                { value: "false", label: "No" },
              ]}
              description="Whether to normalize features before training"
            />

            <button
              onClick={handleTrain}
              disabled={trainMutation.isPending}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {trainMutation.isPending ? "Training..." : "Train Model"}
            </button>
          </div>
        ),

        code: (
          <CodeDisplay
            code={generateCode()}
            language="python"
            showLineNumbers={true}
          />
        ),

        visualization: (
          <div className="space-y-6">
            {trainMutation.isPending && (
              <LoadingSpinner message="Training Naive Bayes classifier..." />
            )}

            {trainMutation.isError && (
              <ErrorDisplay
                error={trainMutation.error as Error}
                title="Training Failed"
                retry={handleTrain}
              />
            )}

            {results && (
              <>
                {/* Metrics Display */}
                <ResultsPanel
                  title="Performance Metrics"
                  metrics={{
                    Accuracy: `${(results.metrics.accuracy * 100).toFixed(2)}%`,
                    Precision: `${(results.metrics.precision * 100).toFixed(2)}%`,
                    Recall: `${(results.metrics.recall * 100).toFixed(2)}%`,
                    "F1 Score": `${(results.metrics.f1_score * 100).toFixed(2)}%`,
                    "Execution Time": `${results.execution_time_ms.toFixed(2)}ms`,
                  }}
                />

                {/* Confusion Matrix */}
                {results.visualization_data?.confusion_matrix && (
                  <div className="mt-6">
                    <h3 className="text-lg font-semibold mb-4">Confusion Matrix</h3>
                    <ConfusionMatrix
                      matrix={results.visualization_data.confusion_matrix}
                      labels={results.visualization_data.class_names || []}
                    />
                  </div>
                )}

                {/* Class Probabilities */}
                {results.visualization_data?.probability_distributions && (
                  <div className="mt-6">
                    <h3 className="text-lg font-semibold mb-4">
                      Sample Probability Distributions
                    </h3>
                    <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg overflow-x-auto">
                      <table className="min-w-full text-sm">
                        <thead>
                          <tr>
                            <th className="px-4 py-2 text-left">Sample</th>
                            <th className="px-4 py-2 text-left">True</th>
                            <th className="px-4 py-2 text-left">Predicted</th>
                            {results.visualization_data.class_names?.map(
                              (name: string, idx: number) => (
                                <th key={idx} className="px-4 py-2 text-left">
                                  P({name})
                                </th>
                              )
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          {results.visualization_data.probability_distributions.samples
                            .slice(0, 10)
                            .map((probs: number[], idx: number) => (
                              <tr key={idx} className="border-t border-gray-200 dark:border-gray-700">
                                <td className="px-4 py-2">{idx + 1}</td>
                                <td className="px-4 py-2">
                                  {results.visualization_data.class_names?.[
                                    results.visualization_data.probability_distributions.true_labels[idx]
                                  ] || results.visualization_data.probability_distributions.true_labels[idx]}
                                </td>
                                <td className="px-4 py-2 font-medium">
                                  {results.visualization_data.class_names?.[
                                    results.visualization_data.probability_distributions.predicted_labels[idx]
                                  ] || results.visualization_data.probability_distributions.predicted_labels[idx]}
                                </td>
                                {probs.map((prob: number, probIdx: number) => (
                                  <td
                                    key={probIdx}
                                    className="px-4 py-2"
                                    style={{
                                      backgroundColor: `rgba(59, 130, 246, ${prob * 0.3})`,
                                    }}
                                  >
                                    {(prob * 100).toFixed(1)}%
                                  </td>
                                ))}
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Class Prior Probabilities */}
                {results.visualization_data?.class_priors && (
                  <div className="mt-6">
                    <h3 className="text-lg font-semibold mb-4">
                      Class Prior Probabilities
                    </h3>
                    <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {results.visualization_data.class_priors.map(
                          (prior: number, idx: number) => (
                            <div
                              key={idx}
                              className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow"
                            >
                              <div className="text-sm text-gray-500 dark:text-gray-400">
                                {results.visualization_data.class_names?.[idx] || `Class ${idx}`}
                              </div>
                              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                                {(prior * 100).toFixed(2)}%
                              </div>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Feature Importance */}
                {results.visualization_data?.feature_importance && (
                  <div className="mt-6">
                    <h3 className="text-lg font-semibold mb-4">
                      Feature Importance (Discriminative Power)
                    </h3>
                    <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                      {results.visualization_data.feature_importance.importance_scores.map(
                        (score: number, idx: number) => (
                          <div key={idx} className="mb-3">
                            <div className="flex justify-between text-sm mb-1">
                              <span>Feature {idx + 1}</span>
                              <span className="font-medium">
                                {(score * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                              <div
                                className="bg-blue-600 h-2 rounded-full transition-all"
                                style={{ width: `${score * 100}%` }}
                              />
                            </div>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}

                {/* Theory Section */}
                <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <h4 className="font-semibold mb-2">About Naive Bayes</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    Naive Bayes is a probabilistic classifier that applies Bayes' theorem
                    with the "naive" assumption that features are independent. Despite this
                    simplification, it often performs surprisingly well in practice,
                    especially for text classification and spam filtering.
                  </p>
                </div>
              </>
            )}
          </div>
        ),

        theory: (
          <div className="prose dark:prose-invert max-w-none">
            <h3>Algorithm Theory</h3>
            <p>
              Naive Bayes is based on Bayes' theorem, which describes the probability
              of an event based on prior knowledge of conditions related to the event.
              The classifier calculates the probability of each class given the features,
              and selects the class with the highest probability.
            </p>

            <h4>Key Concepts</h4>
            <ul>
              <li>
                <strong>Bayes' Theorem:</strong> P(Class|Features) = P(Features|Class) ×
                P(Class) / P(Features)
              </li>
              <li>
                <strong>Independence Assumption:</strong> Features are assumed to be
                conditionally independent given the class
              </li>
              <li>
                <strong>Gaussian Distribution:</strong> Continuous features are assumed
                to follow a normal distribution within each class
              </li>
            </ul>

            <h4>Advantages</h4>
            <ul>
              <li>Fast training and prediction (O(n×d) time complexity)</li>
              <li>Works well with high-dimensional data</li>
              <li>Requires small amount of training data</li>
              <li>Not sensitive to irrelevant features</li>
              <li>Provides probability estimates</li>
            </ul>

            <h4>Disadvantages</h4>
            <ul>
              <li>Assumes feature independence (rarely true in practice)</li>
              <li>Sensitive to feature scaling for Gaussian variant</li>
              <li>Can be outperformed by more complex models</li>
              <li>May produce biased probability estimates</li>
            </ul>

            <h4>Use Cases</h4>
            <ul>
              <li>Spam email filtering</li>
              <li>Document classification</li>
              <li>Sentiment analysis</li>
              <li>Medical diagnosis</li>
              <li>Real-time prediction (due to fast training/inference)</li>
            </ul>
          </div>
        ),
      }}
    />
  );
}

export default NaiveBayes;
