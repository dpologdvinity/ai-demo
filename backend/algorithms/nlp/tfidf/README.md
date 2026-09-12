# TF-IDF (Term Frequency-Inverse Document Frequency)

## Overview

TF-IDF is a statistical measure used to evaluate the importance of words in a collection of documents. This implementation provides a complete backend API for computing TF-IDF scores, extracting key terms, and visualizing results.

## Algorithm Details

- **Name**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Category**: Natural Language Processing (NLP)
- **Difficulty**: Beginner
- **Library**: scikit-learn TfidfVectorizer
- **Complexity**: 
  - Time: O(n*m) where n=documents, m=terms
  - Space: O(n*m)

## Implementation Files

### 1. `schema.py`
Defines request and response models using Pydantic:
- `TFIDFRequest`: Input parameters for TF-IDF computation
- `TFIDFResponse`: Output with TF-IDF matrix, top terms, and visualization data
- `DocumentTerms`: Top terms for individual documents

### 2. `data.py`
Provides sample corpus and data utilities:
- `get_sample_corpus()`: Returns 20 diverse news articles across 7 topics
- `get_corpus_info()`: Returns corpus metadata
- `validate_custom_corpus()`: Validates user-provided documents
- `get_dataset_info()`: Returns dataset statistics for API

### 3. `model.py`
Core TF-IDF implementation:
- `TFIDFModel`: Main model class with scikit-learn TfidfVectorizer
- `fit_transform()`: Fit vectorizer and transform documents
- `get_top_terms_per_document()`: Extract top N terms for each document
- `get_top_terms_global()`: Extract most important terms across all documents
- `get_heatmap_data()`: Prepare data for heatmap visualization
- `train_tfidf()`: Main training function called by API endpoint

### 4. `__init__.py`
Package exports for easy importing

## API Endpoints

### POST `/api/nlp/tfidf/train`
Compute TF-IDF scores for document corpus.

**Request Body**:
```json
{
  "max_features": 100,
  "ngram_range": [1, 1],
  "min_df": 1,
  "max_df": 1.0,
  "use_idf": true,
  "custom_documents": null,
  "normalize": true
}
```

**Response**:
```json
{
  "success": true,
  "metrics": {
    "vocabulary_size": 100,
    "total_documents": 20,
    "avg_terms_per_doc": 45.2,
    "sparsity": 0.52
  },
  "feature_names": ["artificial", "intelligence", "learning", ...],
  "tfidf_matrix": [[0.42, 0.31, ...], ...],
  "top_terms_per_doc": [...],
  "top_terms_global": [...],
  "heatmap_data": {...},
  "visualization_data": {...},
  "execution_time_ms": 125.4
}
```

### GET `/api/nlp/tfidf/info`
Get algorithm metadata and dataset information.

## Parameters

1. **max_features** (10-500, default: 100)
   - Maximum number of features/terms to extract
   
2. **ngram_range** ((1,1), (1,2), (1,3), default: (1,1))
   - N-gram range for feature extraction
   - (1,1) = unigrams only
   - (1,2) = unigrams + bigrams
   - (1,3) = unigrams + bigrams + trigrams

3. **min_df** (1-10, default: 1)
   - Minimum document frequency
   - Terms must appear in at least this many documents

4. **max_df** (0.5-1.0, default: 1.0)
   - Maximum document frequency as proportion
   - Filter out terms appearing in too many documents

5. **use_idf** (boolean, default: true)
   - Enable inverse document frequency weighting
   - When false, becomes simple term frequency

6. **normalize** (boolean, default: true)
   - Normalize TF-IDF vectors to unit length

## Sample Corpus

The implementation includes 20 sample documents covering:
- Technology & AI (3 docs)
- Healthcare & Medicine (3 docs)
- Finance & Economics (3 docs)
- Environment & Climate (3 docs)
- Sports & Athletics (3 docs)
- Entertainment & Media (3 docs)
- Education & Learning (2 docs)

## Visualizations

The API returns data for three main visualizations:

1. **Heatmap**: TF-IDF scores matrix (documents × terms)
   - Shows importance of terms across documents
   - Limited to top 30 terms and 20 documents for clarity

2. **Top Terms Chart**: Bar chart of most important terms globally
   - Shows mean TF-IDF scores across all documents
   - Highlights distinctive vocabulary

3. **Document Statistics**: Term count per document
   - Shows vocabulary richness of each document

## Use Cases

- Document classification
- Information retrieval and search engines
- Keyword extraction
- Text mining and analysis
- Content recommendation systems

## Testing

Run the test script to verify implementation:
```bash
cd backend
python3 test_tfidf.py
```

## Example Usage

```python
from algorithms.nlp.tfidf import TFIDFRequest, train_tfidf

# Create request
request = TFIDFRequest(
    max_features=100,
    ngram_range=(1, 2),  # Unigrams + bigrams
    min_df=2,
    use_idf=True
)

# Compute TF-IDF
response = train_tfidf(request)

# Access results
print(f"Vocabulary size: {response.metrics['vocabulary_size']}")
print(f"Top term: {response.top_terms_global[0]['term']}")
```

## Metadata Registration

The algorithm is registered in `/api/routes/nlp.py` with complete metadata including:
- Parameters and their constraints
- Complexity analysis
- Use cases and applications
- Pros and cons
- Related algorithms
- Educational theory

## Notes

- Stop words (common English words) are automatically filtered
- All text is lowercased during processing
- Sparse matrices are used internally for efficiency
- Results are deterministic (no randomness involved)
