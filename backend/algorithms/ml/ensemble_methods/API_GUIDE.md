# Ensemble Methods API Guide for Frontend Developers

## Quick Start

### Endpoint: Train Ensemble Model
**POST** `/api/ml/ensemble-methods/train`

### Basic Request Example
```json
{
  "method": "voting",
  "n_estimators": 10,
  "base_model": "decision_tree",
  "max_samples": 0.8,
  "learning_rate": 1.0,
  "test_size": 0.3,
  "random_state": 42,
  "dataset_name": "wine",
  "normalize": true
}
```

### Response Structure
```json
{
  "success": true,
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.93,
    "f1_score": 0.935,
    "train_accuracy": 0.98,
    "test_accuracy": 0.95
  },
  "predictions": [0, 1, 2, ...],
  "actual": [0, 1, 2, ...],
  "visualization_data": {
    "performance_comparison": {
      "single_models": [
        {
          "model_name": "Decision Tree",
          "accuracy": 0.85,
          "precision": 0.84,
          "recall": 0.83,
          "f1_score": 0.835
        },
        ...
      ],
      "ensemble_metrics": {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1_score": 0.935
      }
    },
    "feature_importance": {
      "feature1": 0.25,
      "feature2": 0.18,
      ...
    },
    "confusion_matrix": [[45, 2, 0], [1, 38, 2], [0, 1, 44]],
    "class_labels": ["Class 0", "Class 1", "Class 2"],
    "diversity_metrics": {
      "disagreement": 0.15,
      "avg_correlation": 0.65,
      "q_statistic": 0.45
    },
    "voting_patterns": {
      "sample_indices": [0, 1, 2, ...],
      "model_predictions": [[0, 0, 1, 0], [1, 1, 1, 1], ...],
      "ensemble_predictions": [0, 1, ...],
      "actual_labels": [0, 1, ...],
      "agreement_scores": [0.75, 1.0, ...]
    },
    "individual_predictions": {
      "Decision Tree": [0, 1, 2, ...],
      "SVM": [0, 1, 2, ...],
      "KNN": [0, 1, 2, ...],
      "Logistic Regression": [0, 1, 2, ...]
    }
  },
  "execution_time_ms": 234.5,
  "parameters": { ... },
  "model_info": {
    "method": "voting",
    "n_estimators": 10,
    "base_model": "decision_tree",
    "n_base_models": 4,
    "n_features": 13,
    "n_classes": 3,
    "dataset": "wine"
  },
  "message": "Successfully trained voting ensemble with 10 estimators"
}
```

## Request Parameters

### method (required)
Type: `string`
Options: `"bagging"`, `"boosting"`, `"stacking"`, `"voting"`, `"all"`
Default: `"voting"`

**Description**: Ensemble method to use
- **bagging**: Bootstrap aggregating (Random Forest style)
- **boosting**: Gradient Boosting (sequential error correction)
- **stacking**: Meta-learning with multiple layers
- **voting**: Soft voting (probability averaging)
- **all**: Compare all methods (future feature)

### n_estimators
Type: `integer`
Range: 3-100
Default: `10`

**Description**: Number of base models in the ensemble. More models generally improve performance but increase computation time.

### base_model
Type: `string`
Options: `"decision_tree"`, `"svm"`, `"knn"`, `"logistic"`
Default: `"decision_tree"`

**Description**: Type of base estimator (primarily for Bagging)
- **decision_tree**: Fast, interpretable
- **svm**: Good for complex boundaries
- **knn**: Simple, instance-based
- **logistic**: Linear, probabilistic

### max_samples
Type: `float`
Range: 0.5-1.0
Default: `0.8`

**Description**: Ratio of samples to draw for each base model (Bagging only). Lower values increase diversity but may reduce individual model accuracy.

### learning_rate
Type: `float`
Range: 0.1-2.0
Default: `1.0`

**Description**: Learning rate for Boosting. Lower values require more estimators but may generalize better.

### test_size
Type: `float`
Range: 0.1-0.5
Default: `0.3`

**Description**: Proportion of data to use for testing.

### random_state
Type: `integer`
Default: `42`

**Description**: Random seed for reproducibility.

### dataset_name
Type: `string`
Options: `"wine"`, `"breast_cancer"`, `"iris"`, `"digits"`
Default: `"wine"`

**Description**: Dataset to use for training.

### normalize
Type: `boolean`
Default: `true`

**Description**: Whether to normalize features before training.

## Visualization Components

### 1. Performance Comparison Bar Chart
**Data**: `visualization_data.performance_comparison`

Show bars comparing:
- Single model accuracies (Decision Tree, SVM, KNN, Logistic Regression)
- Ensemble accuracy (highlight in different color)

```javascript
const chartData = response.visualization_data.performance_comparison;
// single_models array + ensemble_metrics for comparison
```

### 2. Individual Model Predictions Heatmap
**Data**: `visualization_data.individual_predictions`

Rows: Samples (limited to 50 for visualization)
Columns: Models (Decision Tree, SVM, KNN, Logistic Regression, Ensemble)
Colors: Predicted classes
Annotations: Mark correct/incorrect predictions

```javascript
const predictions = response.visualization_data.individual_predictions;
// Object with model names as keys, prediction arrays as values
```

### 3. Ensemble Voting Visualization
**Data**: `visualization_data.voting_patterns`

Show for each sample:
- Predictions from each model
- Final ensemble prediction
- Actual label
- Agreement score (color intensity)

```javascript
const votingData = response.visualization_data.voting_patterns;
// sample_indices, model_predictions, ensemble_predictions, actual_labels, agreement_scores
```

### 4. Diversity Metrics Display
**Data**: `visualization_data.diversity_metrics`

Show three gauges/cards:
- **Disagreement** (0-1): Higher = more diversity
- **Avg Correlation** (-1 to 1): Lower = more diversity
- **Q-Statistic** (-1 to 1): Negative = diverse, Positive = similar

```javascript
const diversity = response.visualization_data.diversity_metrics;
// disagreement, avg_correlation, q_statistic
```

### 5. Confusion Matrix
**Data**: `visualization_data.confusion_matrix` + `visualization_data.class_labels`

Standard confusion matrix visualization with class labels.

```javascript
const cm = response.visualization_data.confusion_matrix;
const labels = response.visualization_data.class_labels;
```

### 6. Feature Importance Chart
**Data**: `visualization_data.feature_importance`

Horizontal bar chart sorted by importance.

```javascript
const importance = response.visualization_data.feature_importance;
// Object with feature names as keys, importance scores as values
```

### 7. Model Agreement/Disagreement Chart
**Data**: `visualization_data.voting_patterns.agreement_scores`

Histogram showing distribution of agreement scores:
- 1.0 = All models agree (very confident)
- 0.5 = Models split (uncertain)
- Distribution shape indicates ensemble confidence

```javascript
const agreementScores = response.visualization_data.voting_patterns.agreement_scores;
// Array of floats from 0 to 1
```

## Example Frontend Code

### Fetch and Display
```typescript
async function trainEnsemble(params: EnsembleParams) {
  try {
    const response = await fetch('/api/ml/ensemble-methods/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    
    const data = await response.json();
    
    if (data.success) {
      // Display metrics
      displayMetrics(data.metrics);
      
      // Render performance comparison
      renderPerformanceChart(data.visualization_data.performance_comparison);
      
      // Show diversity metrics
      displayDiversityMetrics(data.visualization_data.diversity_metrics);
      
      // Render voting patterns
      renderVotingHeatmap(data.visualization_data.voting_patterns);
      
      // Show confusion matrix
      renderConfusionMatrix(
        data.visualization_data.confusion_matrix,
        data.visualization_data.class_labels
      );
    }
  } catch (error) {
    console.error('Training failed:', error);
  }
}
```

### Performance Comparison Chart (Recharts)
```typescript
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

function PerformanceComparison({ data }) {
  const chartData = [
    ...data.single_models.map(m => ({
      name: m.model_name,
      accuracy: m.accuracy,
      type: 'single'
    })),
    {
      name: 'Ensemble',
      accuracy: data.ensemble_metrics.accuracy,
      type: 'ensemble'
    }
  ];
  
  return (
    <BarChart width={600} height={400} data={chartData}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis domain={[0, 1]} />
      <Tooltip />
      <Legend />
      <Bar 
        dataKey="accuracy" 
        fill={(entry) => entry.type === 'ensemble' ? '#22c55e' : '#3b82f6'}
      />
    </BarChart>
  );
}
```

### Voting Heatmap (D3 or custom)
```typescript
function VotingHeatmap({ data }) {
  const { model_predictions, ensemble_predictions, actual_labels, agreement_scores } = data;
  
  return (
    <div className="voting-heatmap">
      {model_predictions.map((samplePreds, idx) => (
        <div key={idx} className="sample-row">
          {samplePreds.map((pred, modelIdx) => (
            <div 
              key={modelIdx}
              className={`cell ${pred === actual_labels[idx] ? 'correct' : 'incorrect'}`}
              style={{ opacity: agreement_scores[idx] }}
            >
              {pred}
            </div>
          ))}
          <div className="ensemble-cell">{ensemble_predictions[idx]}</div>
          <div className="actual-cell">{actual_labels[idx]}</div>
        </div>
      ))}
    </div>
  );
}
```

## Get Algorithm Info
**GET** `/api/ml/ensemble-methods/info`

Returns algorithm metadata for display in UI:
```json
{
  "metadata": {
    "id": "ensemble-methods",
    "name": "Ensemble Methods",
    "slug": "ensemble-methods",
    "category": "ml",
    "description": "Combine multiple models to improve prediction accuracy and robustness",
    "difficulty": "intermediate",
    "tags": ["ml", "ensemble", "bagging", "boosting", "stacking", "model-combination"],
    "use_cases": [...],
    "complexity": {
      "time": "O(n_estimators*base_complexity)",
      "space": "O(n_estimators*base_space)"
    },
    "parameters": [...],
    "theory": "...",
    "pros": [...],
    "cons": [...],
    "related_algorithms": [...]
  },
  "available_datasets": ["wine", "breast_cancer", "iris", "digits"]
}
```

## Error Handling

### Validation Errors (400)
```json
{
  "detail": "n_estimators must be between 3 and 100"
}
```

### Training Errors (500)
```json
{
  "detail": "Training failed: Insufficient samples for training"
}
```

## Performance Notes

- **Bagging**: Fast parallel training
- **Boosting**: Sequential, slower but more accurate
- **Stacking**: Slowest (cross-validation), highest potential accuracy
- **Voting**: Moderate speed, good balance

Typical execution times:
- Wine dataset: 200-500ms
- Breast Cancer: 150-400ms
- Iris: 100-300ms
- Digits: 500-1500ms (more features)

## Tips for UI/UX

1. **Show method comparison**: Let users see voting vs bagging vs boosting
2. **Highlight ensemble improvement**: Show % improvement over best single model
3. **Diversity explanation**: Higher diversity = more robust but sometimes lower individual accuracy
4. **Agreement visualization**: High agreement samples = confident predictions
5. **Real-time updates**: Show training progress if possible
6. **Parameter hints**: Suggest n_estimators based on dataset size
7. **Interactive exploration**: Click on samples to see which models got them right/wrong

## Common Use Patterns

### 1. Quick comparison
```json
{
  "method": "voting",
  "n_estimators": 10,
  "dataset_name": "wine"
}
```

### 2. High accuracy (slower)
```json
{
  "method": "boosting",
  "n_estimators": 50,
  "learning_rate": 0.5,
  "dataset_name": "wine"
}
```

### 3. Fast but diverse
```json
{
  "method": "bagging",
  "n_estimators": 20,
  "max_samples": 0.7,
  "dataset_name": "wine"
}
```

### 4. Maximum performance
```json
{
  "method": "stacking",
  "n_estimators": 30,
  "dataset_name": "wine"
}
```
