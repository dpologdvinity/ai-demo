# Text Classification Algorithm

## Overview

Text Classification is a supervised machine learning algorithm that assigns predefined categories to text documents. This implementation uses TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction and supports multiple classifiers.

## Features

- **Multiple Classifiers**: Naive Bayes, Logistic Regression, Support Vector Machine
- **TF-IDF Vectorization**: Configurable n-grams and feature count
- **Comprehensive Metrics**: Per-class and overall performance metrics
- **Feature Analysis**: Top discriminative features for each class
- **Confusion Matrix**: Visual representation of classification performance
- **Custom Datasets**: Support for user-provided text and labels

## Algorithm Details

### Classifiers

1. **Naive Bayes (MultinomialNB)**
   - Probabilistic classifier based on Bayes' theorem
   - Assumes feature independence
   - Fast and effective for text classification
   - Works well with small datasets

2. **Logistic Regression**
   - Linear model with multinomial classification
   - Interpretable feature weights
   - Good generalization performance
   - Supports regularization

3. **Support Vector Machine (LinearSVC)**
   - Finds optimal hyperplane to separate classes
   - Effective in high-dimensional spaces
   - Memory efficient with linear kernel
   - Good for complex classification tasks

### Feature Extraction

- **TF-IDF Vectorization**:
  - Term Frequency: How often a term appears in a document
  - Inverse Document Frequency: How rare a term is across all documents
  - Configurable n-gram range (unigrams, bigrams, trigrams)
  - Stop word removal
  - Vocabulary size control

## Dataset

The default dataset contains 100 labeled text samples across 5 categories:

- **Technology** (20 samples): Tech news, product reviews, innovations
- **Sports** (20 samples): Game results, athlete profiles, sporting events
- **Politics** (20 samples): Political news, policy, elections
- **Entertainment** (20 samples): Movies, TV, music, celebrities
- **Business** (20 samples): Financial markets, corporate news, economics

Each category has diverse examples covering various topics within the domain.

## API Usage

### Training Endpoint

```http
POST /api/nlp/text-classification/train
Content-Type: application/json

{
  "classifier_type": "naive_bayes",
  "max_features": 1000,
  "test_size": 0.2,
  "ngram_range": [1, 2]
}
```

### Custom Dataset

```http
POST /api/nlp/text-classification/train
Content-Type: application/json

{
  "classifier_type": "logistic_regression",
  "max_features": 500,
  "test_size": 0.25,
  "ngram_range": [1, 2],
  "use_custom_dataset": true,
  "custom_texts": [
    "This is a positive review...",
    "This is negative feedback..."
  ],
  "custom_labels": [
    "positive",
    "negative"
  ]
}
```

### Info Endpoint

```http
GET /api/nlp/text-classification/info
```

## Response Format

```json
{
  "success": true,
  "predictions": [
    {
      "text": "Sample text...",
      "true_label": "technology",
      "predicted_label": "technology",
      "confidence": 0.8542,
      "probabilities": {
        "technology": 0.8542,
        "sports": 0.0234,
        "politics": 0.0512,
        "entertainment": 0.0398,
        "business": 0.0314
      }
    }
  ],
  "confusion_matrix": [[18, 0, 1, 0, 1], ...],
  "class_names": ["business", "entertainment", "politics", "sports", "technology"],
  "class_metrics": [
    {
      "class_name": "technology",
      "precision": 0.9500,
      "recall": 0.9000,
      "f1_score": 0.9245,
      "support": 20
    }
  ],
  "overall_metrics": {
    "accuracy": 0.8900,
    "macro_precision": 0.8850,
    "macro_recall": 0.8800,
    "macro_f1": 0.8825,
    "weighted_precision": 0.8920,
    "weighted_recall": 0.8900,
    "weighted_f1": 0.8910
  },
  "top_features_per_class": {
    "technology": [
      {"feature": "software", "weight": 2.3456},
      {"feature": "algorithm", "weight": 2.1234}
    ]
  },
  "execution_time_ms": 234.56,
  "model_info": {
    "classifier_type": "naive_bayes",
    "vectorizer": "TF-IDF",
    "vocabulary_size": 987,
    "train_size": 80,
    "test_size": 20,
    "num_classes": 5
  }
}
```

## Metrics Explanation

### Per-Class Metrics

- **Precision**: Proportion of positive predictions that are correct
  - `precision = true_positives / (true_positives + false_positives)`
  - Answers: "When we predict this class, how often are we correct?"

- **Recall**: Proportion of actual positives correctly identified
  - `recall = true_positives / (true_positives + false_negatives)`
  - Answers: "Of all instances of this class, how many did we find?"

- **F1-Score**: Harmonic mean of precision and recall
  - `f1 = 2 * (precision * recall) / (precision + recall)`
  - Balanced measure of both precision and recall

- **Support**: Number of actual occurrences of the class in the test set

### Overall Metrics

- **Accuracy**: Overall proportion of correct predictions
- **Macro Average**: Unweighted mean of per-class metrics
- **Weighted Average**: Mean of per-class metrics weighted by support

## Visualization Data

The response includes visualization-ready data:

1. **Confusion Matrix**: Both raw counts and normalized percentages
2. **Confidence Distribution**: Statistics on prediction confidence
3. **Sample Predictions Table**: Detailed view of predictions
4. **Top Features**: Most discriminative words per class

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| classifier_type | select | naive_bayes | naive_bayes, logistic_regression, svm | ML model type |
| max_features | range | 1000 | 100-5000 | Max TF-IDF features |
| test_size | range | 0.2 | 0.1-0.4 | Test set proportion |
| ngram_range | select | (1,2) | (1,1), (1,2), (1,3) | N-gram range |
| use_custom_dataset | boolean | false | - | Use custom data |
| custom_texts | array | null | - | Custom text samples |
| custom_labels | array | null | - | Corresponding labels |

## Use Cases

1. **Sentiment Analysis**: Classify customer reviews as positive/negative/neutral
2. **Spam Detection**: Identify spam vs. legitimate messages
3. **Topic Classification**: Categorize news articles by topic
4. **Intent Detection**: Determine user intent in chatbots
5. **Document Categorization**: Organize documents by content type

## Complexity

- **Time Complexity**: O(n × m)
  - n = number of documents
  - m = average number of features per document
  
- **Space Complexity**: O(vocab_size × num_classes)
  - Stores model parameters for all features and classes

## Implementation Notes

- Uses stratified train-test split to maintain class balance
- Applies English stop word removal
- Normalizes text (lowercase, unicode handling)
- Limits text length to 1000 characters
- Requires minimum 10 samples and 2 classes for custom datasets

## Dependencies

- scikit-learn: Machine learning models and metrics
- numpy: Numerical computations
- typing: Type hints

## Example Code

```python
from algorithms.nlp.text_classification import (
    TextClassificationModel,
    TextClassificationParameters
)

# Create parameters
params = TextClassificationParameters(
    classifier_type="logistic_regression",
    max_features=1000,
    test_size=0.2,
    ngram_range=(1, 2)
)

# Train model
model = TextClassificationModel()
result = model.train(params)

# Check results
if result.success:
    print(f"Accuracy: {result.overall_metrics['accuracy']:.2%}")
    print(f"Classes: {result.class_names}")
    
    # View top features for a class
    for class_name, features in result.top_features_per_class.items():
        print(f"\nTop features for {class_name}:")
        for feature in features[:5]:
            print(f"  - {feature.feature}: {feature.weight:.4f}")
```

## Error Handling

The implementation includes comprehensive error handling:

- Parameter validation (types, ranges)
- Dataset validation (minimum samples, class count)
- Text preprocessing (empty texts, length limits)
- Model training errors
- Prediction errors

All errors return a response with `success=false` and descriptive error messages.
