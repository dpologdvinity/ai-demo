# Topic Modeling (LDA) Implementation

## Overview

This implementation provides a complete Topic Modeling solution using Latent Dirichlet Allocation (LDA) for discovering abstract topics in document collections.

## Files Created

```
backend/algorithms/nlp/topic_modeling/
├── __init__.py          # Module exports
├── schema.py            # Request/Response models (Pydantic schemas)
├── data.py             # Sample documents corpus and data utilities
└── model.py            # LDA implementation using scikit-learn
```

## API Endpoints

### 1. Train Topic Model
**POST** `/api/nlp/topic-modeling/train`

Trains an LDA model to discover topics in a document corpus.

**Request Body:**
```json
{
  "n_topics": 5,
  "max_iterations": 100,
  "alpha": "auto",
  "beta": "auto",
  "min_df": 2,
  "max_df": 0.95,
  "use_custom_documents": false,
  "custom_documents": null
}
```

**Response:**
```json
{
  "success": true,
  "topics": [
    {
      "topic_id": 0,
      "top_words": [
        {"word": "technology", "weight": 0.025},
        {"word": "data", "weight": 0.022}
      ],
      "keywords": "technology, data, computing, software, applications"
    }
  ],
  "document_topics": [
    {
      "document_id": 0,
      "document_preview": "Artificial intelligence and machine learning...",
      "dominant_topic": 0,
      "topic_distribution": [0.85, 0.05, 0.04, 0.03, 0.03]
    }
  ],
  "coherence_score": 0.9800,
  "perplexity": 301.51,
  "execution_time_ms": 2722.53,
  "metrics": {
    "n_topics": 5,
    "n_documents": 60,
    "vocabulary_size": 119
  },
  "visualization_data": {
    "word_clouds": [...],
    "topic_keywords": [...],
    "heatmap": {...},
    "documents_by_topic": {...}
  }
}
```

### 2. Get Algorithm Info
**GET** `/api/nlp/topic-modeling/info`

Returns algorithm metadata, parameters, dataset info, and usage guidelines.

**Response:**
```json
{
  "metadata": {
    "name": "Topic Modeling (LDA)",
    "description": "Discover abstract topics in document collections...",
    "difficulty": "intermediate",
    "tags": ["nlp", "topic-modeling", "lda", "unsupervised"]
  },
  "dataset": {
    "num_documents": 60,
    "topics_covered": [
      "Technology & Computing",
      "Sports & Athletics",
      "Politics & Government",
      "Science & Research",
      "Entertainment & Media",
      "Business & Finance"
    ]
  }
}
```

## Parameters

### n_topics (2-20, default: 5)
Number of topics to discover in the corpus. Higher values create more granular topics.

### max_iterations (20-500, default: 100)
Maximum LDA training iterations. More iterations = better convergence but slower training.

### alpha (default: "auto")
Document-topic density parameter. Controls how many topics each document contains.
- "auto": 1/n_topics (recommended)
- 0.1: Sparse (fewer topics per document)
- 0.5: Medium
- 1.0: Dense (more topics per document)

### beta (default: "auto")
Topic-word density parameter. Controls vocabulary size per topic.
- "auto": 1/n_topics (recommended)
- 0.01: Sparse (fewer words per topic)
- 0.1: Medium
- 1.0: Dense (more words per topic)

### min_df (1-10, default: 2)
Minimum document frequency. Terms must appear in at least this many documents.

### max_df (0.5-1.0, default: 0.95)
Maximum document frequency. Terms appearing in more than this proportion are filtered (removes common words).

## Dataset

### Default Corpus
- **60 documents** across 6 diverse topics
- **Topics**: Technology, Sports, Politics, Science, Entertainment, Business
- **~1,166 total words** (average ~19.4 words per document)
- Real-world style content suitable for demonstration

### Custom Documents
You can provide custom documents by setting:
```json
{
  "use_custom_documents": true,
  "custom_documents": [
    "Your first document...",
    "Your second document...",
    "..."
  ]
}
```
**Note**: Minimum 5 documents required for meaningful topic modeling.

## Visualizations

### 1. Word Clouds
One word cloud per topic showing the most important words sized by weight.

### 2. Topic-Word Heatmap
Matrix visualization showing word importance across topics.
- Rows: Topics
- Columns: Top words
- Values: Normalized probabilities

### 3. Top Keywords Bar Chart
Bar charts showing the top 10 keywords per topic with weights.

### 4. Document-Topic Distribution
Shows how topics are distributed across sample documents.

### 5. Documents by Topic
Groups documents by their dominant topic for exploration.

## Metrics

### Coherence Score (0-1, higher is better)
Measures topic quality based on word co-occurrence patterns. Higher scores indicate more interpretable, coherent topics.

### Perplexity (lower is better)
Measures how well the model fits the data. Lower perplexity indicates better model fit.

### Vocabulary Size
Number of unique terms in the vocabulary after filtering.

## Example Usage

### Python
```python
from algorithms.nlp.topic_modeling import TopicModelingModel, TopicModelingParameters

# Create parameters
params = TopicModelingParameters(
    n_topics=5,
    max_iterations=100,
    alpha="auto",
    beta="auto",
    min_df=2,
    max_df=0.95
)

# Train model
model = TopicModelingModel()
result = model.train(params)

# Access results
if result.success:
    print(f"Discovered {len(result.topics)} topics")
    print(f"Coherence: {result.coherence_score:.4f}")
    
    for topic in result.topics:
        print(f"Topic {topic.topic_id}: {topic.keywords}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/nlp/topic-modeling/train" \
  -H "Content-Type: application/json" \
  -d '{
    "n_topics": 5,
    "max_iterations": 100,
    "alpha": "auto",
    "beta": "auto"
  }'
```

### JavaScript (fetch)
```javascript
const response = await fetch('http://localhost:8000/api/nlp/topic-modeling/train', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    n_topics: 5,
    max_iterations: 100,
    alpha: 'auto',
    beta: 'auto'
  })
});

const result = await response.json();
console.log('Topics:', result.topics);
console.log('Coherence:', result.coherence_score);
```

## Algorithm Details

### Theory
Latent Dirichlet Allocation (LDA) is a generative probabilistic model that:
1. Assumes documents are mixtures of topics
2. Each topic is a distribution over words
3. Uses Bayesian inference to discover hidden topic structure

### Generative Process
1. For each document, choose a topic distribution
2. For each word in the document:
   - Choose a topic from the document's distribution
   - Choose a word from that topic's distribution

### Implementation
- **Library**: scikit-learn LatentDirichletAllocation
- **Method**: Online variational Bayes
- **Preprocessing**: Lowercase, stopword removal, CountVectorizer
- **Complexity**: O(iterations × documents × vocabulary)

## Use Cases

1. **Document Clustering**: Automatically group similar documents
2. **Content Recommendation**: Recommend articles based on topic similarity
3. **Trend Analysis**: Discover emerging topics over time
4. **Research Categorization**: Organize academic papers by research themes
5. **Customer Feedback**: Analyze reviews to identify common themes
6. **News Organization**: Categorize news articles by topic

## Performance

- **Training Time**: ~2-3 seconds for 60 documents, 5 topics, 100 iterations
- **Scalability**: Handles 100+ documents efficiently
- **Memory**: O(topics × vocabulary) space complexity

## Testing

Run the comprehensive test suite:
```bash
cd backend
source venv/bin/activate
python test_topic_modeling.py
```

Tests cover:
- Basic training
- Parameter variations
- Document-topic distributions
- Visualization data
- Custom documents
- Metrics calculation

## Integration

### Frontend Integration
1. Call `/api/nlp/topic-modeling/info` to get parameter metadata
2. Render parameter controls based on metadata
3. POST to `/api/nlp/topic-modeling/train` with parameters
4. Render visualizations using `visualization_data`

### Visualization Components Needed
- Word cloud component (accepts word-weight pairs)
- Heatmap component (2D matrix with labels)
- Bar chart component (keywords with weights)
- Distribution chart (topic probabilities per document)
- Document explorer (group documents by topic)

## Dependencies

- scikit-learn >= 1.5.2 (LatentDirichletAllocation, CountVectorizer)
- numpy >= 2.1.1
- pydantic >= 2.9.2

All dependencies are already included in `requirements.txt`.

## Notes

- Minimum 5 documents required for meaningful results
- Quality improves with more documents (50+ recommended)
- Preprocessing removes stopwords and rare/common terms
- Coherence score is simplified; production systems should use gensim's CoherenceModel
- For very large corpora (1000+ documents), consider using gensim for better performance

## Related Algorithms

- TF-IDF: Term weighting for feature extraction
- Word2Vec: Word embeddings for semantic similarity
- Bag of Words: Basic text representation
- NMF: Alternative matrix factorization for topic modeling
