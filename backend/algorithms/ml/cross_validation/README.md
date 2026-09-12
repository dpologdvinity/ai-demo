# Cross-Validation Implementation

## Overview
Implementation of various cross-validation strategies for model evaluation in the AI Algorithms Demo website.

## Algorithm Details
- **Name**: Cross-Validation
- **Slug**: cross-validation
- **Category**: ML (Machine Learning)
- **Difficulty**: Beginner
- **Complexity**: Time O(k*model_training), Space O(n/k)

## Features

### 5 CV Strategies Implemented
1. **K-Fold**: Standard k-fold cross-validation
2. **Stratified K-Fold**: Preserves class distribution in each fold
3. **Shuffle Split**: Random train/test splits
4. **Leave-One-Out**: Each sample as test set once (for small datasets)
5. **Time Series Split**: Respects temporal order

### 4 Models Supported
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### 5 Scoring Metrics
- Accuracy
- F1 Score
- Precision
- Recall
- ROC AUC

### 3 Datasets Available
- Iris (150 samples, 4 features, 3 classes)
- Wine (178 samples, 13 features, 3 classes)
- Breast Cancer (569 samples, 30 features, 2 classes)

## API Endpoints

### POST `/ml/cross-validation/train`
Performs cross-validation on a model.

**Request Body**:
```json
{
  "cv_method": "k_fold",
  "n_splits": 5,
  "model_type": "random_forest",
  "scoring": "accuracy",
  "shuffle": true,
  "dataset": "iris",
  "test_size": 0.2,
  "random_state": 42
}
```

**Response**:
```json
{
  "success": true,
  "metrics": {
    "mean_score": 0.96,
    "std_score": 0.025,
    "min_score": 0.90,
    "max_score": 1.0,
    "mean_train_score": 0.98,
    "std_train_score": 0.01,
    "stability_coefficient": 0.026,
    "scoring_metric": "accuracy"
  },
  "fold_metrics": [...],
  "visualization_data": {
    "fold_performance": [...],
    "score_distribution": [...],
    "train_test_splits": [...],
    "confusion_matrices": [...],
    "class_distributions": [...]
  },
  "execution_time_ms": 125.5,
  "parameters": {...},
  "message": "Successfully completed k_fold cross-validation with 5 folds"
}
```

### GET `/ml/cross-validation/info`
Returns algorithm metadata including parameters, complexity, theory, and use cases.

## Visualization Data

### 1. Fold Performance (Bar Chart)
Per-fold train and test scores for comparison.

### 2. Score Distribution (Box Plot)
Distribution of scores across all folds showing:
- Mean, median, Q1, Q3, min, max
- Separate distributions for train and test scores

### 3. Train/Test Splits Diagram
Visual representation of how data is split across folds.

### 4. Confusion Matrices
Per-fold confusion matrices showing classification performance.

### 5. Class Distribution
Shows class distribution in train and test sets for each fold (useful for stratified CV).

## Key Metrics

### Aggregate Metrics
- **Mean Score**: Average performance across folds
- **Standard Deviation**: Variability of performance
- **Min/Max Scores**: Range of performance
- **Stability Coefficient**: CV (std/mean) - lower is more stable

### Per-Fold Metrics
- Train and test scores
- Train and test set sizes
- Class distribution per fold

## Use Cases
1. **Model Evaluation**: Robust estimate of model performance
2. **Performance Estimation**: Understand model variance
3. **Model Selection**: Compare different models fairly
4. **Preventing Overfitting**: Detect if model generalizes well
5. **Small Dataset Handling**: Maximize use of limited data
6. **Robust Accuracy Measurement**: More reliable than single train/test split

## Files Structure
```
cross_validation/
├── __init__.py          # Module exports
├── schema.py            # Pydantic schemas (171 lines)
├── data.py              # Data loading utilities (139 lines)
├── model.py             # CV implementation (414 lines)
└── README.md            # This file
```

## Testing
All CV methods, model types, and datasets have been tested and validated:
- ✅ K-Fold, Stratified, Shuffle Split, Time Series Split
- ✅ Random Forest, Logistic, SVM, KNN models
- ✅ Iris, Wine, Breast Cancer datasets
- ✅ All scoring metrics
- ✅ Visualization data generation

## Theory
Cross-Validation is a resampling technique that partitions data into complementary subsets for training and validation. Multiple rounds are performed using different partitions, and results are averaged to produce a more reliable estimate of model performance.

This is essential for:
- Detecting overfitting
- Model comparison
- Hyperparameter tuning
- Small dataset scenarios

## Complexity Analysis
- **Time Complexity**: O(k * model_training) where k is the number of folds
- **Space Complexity**: O(n/k) where n is the dataset size

For expensive models or large datasets, consider reducing k or using shuffle split with fewer iterations.
