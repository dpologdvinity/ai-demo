# GloVe Word Embeddings Implementation

This directory contains the implementation of GloVe (Global Vectors for Word Representation) for the AI Algorithms Demo website.

## Overview

GloVe is an unsupervised learning algorithm for obtaining vector representations of words. Unlike Word2Vec which uses local context windows, GloVe is trained on aggregated global word-word co-occurrence statistics from a corpus.

## Files

- `model.py` - Main GloVe model implementation
- `data.py` - Data loading and utility functions for embeddings
- `schema.py` - Pydantic schemas for request/response models
- `__init__.py` - Module exports

## Features

### 1. Word Similarity Queries
Find the most similar words to a query word based on cosine similarity of their embeddings.

```python
from algorithms.nlp.glove import GloVeModel, GloVeParameters

params = GloVeParameters(query_word="king", top_k=10)
model = GloVeModel()
result = model.query(params)

# result.similar_words contains list of similar words with scores
```

### 2. Word Analogies
Solve word analogy problems like "king - man + woman = ?"

```python
params = GloVeParameters(
    analogy_word_a="king",
    analogy_word_b="man",
    analogy_word_c="woman"
)
model = GloVeModel()
result = model.query(params)

# result.analogy_result contains the answer (e.g., "queen")
```

### 3. 2D Visualization
Generate t-SNE projections of word embeddings for visualization.

```python
# result.embeddings_2d contains 2D coordinates for words
# result.visualization_data contains formatted data for frontend
```

### 4. Cosine Similarity Matrix
Compute similarity matrices between selected words for heatmap visualization.

```python
# result.cosine_similarity_matrix contains similarity matrix data
```

## API Endpoints

### POST /nlp/glove/query
Query GloVe embeddings for similar words and analogies.

**Request Body:**
```json
{
  "embedding_dim": 100,
  "top_k": 10,
  "query_word": "king",
  "analogy_word_a": "king",
  "analogy_word_b": "man",
  "analogy_word_c": "woman"
}
```

**Response:**
```json
{
  "success": true,
  "metrics": {
    "vocab_size": 170,
    "embedding_dim": 100,
    "total_parameters": 17000
  },
  "similar_words": [
    {"word": "queen", "similarity": 0.5938},
    {"word": "prince", "similarity": 0.4125}
  ],
  "analogy_result": {
    "query": "king - man + woman",
    "result_word": "queen",
    "similarity": 1.0,
    "top_results": [...]
  },
  "embeddings_2d": [...],
  "cosine_similarity_matrix": {...},
  "vocabulary_sample": [...],
  "execution_time_ms": 250.0
}
```

### GET /nlp/glove/info
Get algorithm metadata and information.

## Demo Implementation

For demonstration purposes, this implementation uses a curated subset of ~170 words with synthetic embeddings that preserve realistic semantic relationships:

- **Semantic groups**: Royalty, gender, geography, technology, science, animals, emotions, etc.
- **Preserved analogies**: 
  - king - man + woman ≈ queen
  - paris - france + germany ≈ berlin
- **Dimensions supported**: 50, 100, 200, 300

### Production Usage

In a production system, you would:

1. Download pre-trained GloVe vectors from Stanford NLP:
   - glove.6B (6 billion tokens)
   - glove.42B (42 billion tokens)
   - glove.840B (840 billion tokens)

2. Load the vectors from text files:
```python
def load_glove_vectors(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector
    return embeddings
```

3. Use the actual pre-trained embeddings for queries

## Algorithm Details

**Time Complexity:**
- Training: O(corpus_size) for computing co-occurrence matrix
- Lookup: O(1) for word vector retrieval
- Similar words: O(vocab_size) for computing similarities

**Space Complexity:**
- O(vocab_size × embedding_dim) for storing word vectors

**Key Parameters:**
- `embedding_dim`: Vector dimension (50, 100, 200, 300)
- `top_k`: Number of similar words to return (5-20)
- `query_word`: Word to find similar words for
- `analogy_word_a/b/c`: Words for analogy A - B + C = ?

## Testing

Run the test script:
```bash
cd /home/kaitlyn/git/ai-demo/backend
PYTHONPATH=. ./venv/bin/python3 test_glove.py
```

## References

- Pennington et al., 2014 - "GloVe: Global Vectors for Word Representation"
- Stanford NLP: https://nlp.stanford.edu/projects/glove/
