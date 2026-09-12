# Topic Modeling (LDA) - Quick Start Guide

## 🚀 Quick Start

### Test the Implementation
```bash
cd backend
source venv/bin/activate
python test_topic_modeling.py
```

### Start the Server
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Test the API
```bash
# Get algorithm info
curl http://localhost:8000/api/nlp/topic-modeling/info | jq

# Train with default parameters
curl -X POST http://localhost:8000/api/nlp/topic-modeling/train \
  -H "Content-Type: application/json" \
  -d '{"n_topics": 5, "max_iterations": 100}' | jq
```

## 📁 File Structure

```
backend/
├── algorithms/nlp/topic_modeling/
│   ├── __init__.py       # Module exports
│   ├── schema.py         # Request/Response models
│   ├── data.py          # Sample documents (60 docs)
│   └── model.py         # LDA implementation
├── api/routes/nlp.py    # API endpoints (updated)
├── test_topic_modeling.py           # Test suite
├── TOPIC_MODELING_README.md         # Full documentation
├── IMPLEMENTATION_SUMMARY.md         # Implementation details
└── QUICK_START_TOPIC_MODELING.md    # This file
```

## 🎯 Key Features

- ✅ **60 Sample Documents** across 6 diverse topics
- ✅ **2-20 Topics** configurable
- ✅ **Alpha/Beta Parameters** for controlling density
- ✅ **Custom Documents** support
- ✅ **Word Clouds** data for each topic
- ✅ **Heatmap** visualization data
- ✅ **Coherence & Perplexity** metrics
- ✅ **Document-Topic Distributions**

## 📊 Sample Request/Response

### Request
```json
{
  "n_topics": 5,
  "max_iterations": 100,
  "alpha": "auto",
  "beta": "auto",
  "min_df": 2,
  "max_df": 0.95
}
```

### Response (Simplified)
```json
{
  "success": true,
  "topics": [
    {
      "topic_id": 0,
      "keywords": "technology, data, computing, software, applications",
      "top_words": [
        {"word": "technology", "weight": 0.025},
        {"word": "data", "weight": 0.022}
      ]
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
  "visualization_data": {
    "word_clouds": [...],
    "topic_keywords": [...],
    "heatmap": {...},
    "documents_by_topic": {...}
  }
}
```

## 🔧 Parameters Quick Reference

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `n_topics` | int | 2-20 | 5 | Number of topics to discover |
| `max_iterations` | int | 20-500 | 100 | LDA training iterations |
| `alpha` | string/float | "auto", 0.1-1.0 | "auto" | Document-topic density |
| `beta` | string/float | "auto", 0.01-1.0 | "auto" | Topic-word density |
| `min_df` | int | 1-10 | 2 | Min documents for term |
| `max_df` | float | 0.5-1.0 | 0.95 | Max document frequency |

## 💡 Common Use Cases

### 1. Explore Default Corpus
```python
from algorithms.nlp.topic_modeling import TopicModelingModel, TopicModelingParameters

params = TopicModelingParameters(n_topics=6)
model = TopicModelingModel()
result = model.train(params)

for topic in result.topics:
    print(f"Topic {topic.topic_id}: {topic.keywords}")
```

### 2. Analyze Custom Documents
```python
custom_docs = [
    "Machine learning is transforming technology...",
    "Sports fans celebrate their teams...",
    # ... at least 5 documents
]

params = TopicModelingParameters(
    n_topics=3,
    use_custom_documents=True,
    custom_documents=custom_docs
)
result = model.train(params)
```

### 3. Fine-tune Parameters
```python
params = TopicModelingParameters(
    n_topics=8,           # More granular topics
    max_iterations=200,   # Better convergence
    alpha=0.1,           # Sparse doc-topic
    beta=0.01,           # Sparse topic-word
    min_df=3,            # Remove rare words
    max_df=0.85          # Remove common words
)
```

## 📈 Metrics Interpretation

### Coherence Score (0-1)
- **> 0.95**: Excellent - Topics are very distinct
- **0.85-0.95**: Good - Topics are coherent
- **0.70-0.85**: Fair - Some topic overlap
- **< 0.70**: Poor - Consider adjusting parameters

### Perplexity (Lower is Better)
- Measures model fit to data
- Lower = better predictions
- Use for comparing models with same data
- Range depends on corpus size

## 🐛 Troubleshooting

### Issue: Low Coherence Score
**Solution**: 
- Increase `n_topics` for more granular topics
- Adjust `min_df` to remove rare words
- Adjust `max_df` to remove common words

### Issue: Topics Too Similar
**Solution**:
- Decrease `n_topics`
- Lower `alpha` for sparser distributions
- Lower `beta` for more focused topics

### Issue: Training Too Slow
**Solution**:
- Decrease `max_iterations`
- Reduce corpus size
- Increase `min_df` to reduce vocabulary

### Issue: Empty Topics
**Solution**:
- Decrease `n_topics`
- Increase `max_df` to include more words
- Check corpus has enough diversity

## 🧪 Testing

### Run All Tests
```bash
python test_topic_modeling.py
```

### Run Specific Test
```python
from test_topic_modeling import test_basic_training
test_basic_training()
```

### Quick Smoke Test
```python
from algorithms.nlp.topic_modeling import TopicModelingModel, TopicModelingParameters

params = TopicModelingParameters(n_topics=3, max_iterations=50)
model = TopicModelingModel()
result = model.train(params)
assert result.success, result.error
print("✓ Working!")
```

## 📚 Documentation

- **Full Documentation**: `TOPIC_MODELING_README.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **API Reference**: FastAPI auto-docs at `/docs` when server running

## 🔗 Related Endpoints

```
GET  /api/nlp/algorithms              # List all NLP algorithms
GET  /api/nlp/topic-modeling/info     # Algorithm metadata
POST /api/nlp/topic-modeling/train    # Train model
GET  /api/nlp/tfidf/info             # Related: TF-IDF
GET  /api/nlp/text-classification/info # Related: Text Classification
```

## 💻 Frontend Integration

### Fetch Metadata
```javascript
const metadata = await fetch('/api/nlp/topic-modeling/info').then(r => r.json());
```

### Train Model
```javascript
const result = await fetch('/api/nlp/topic-modeling/train', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ n_topics: 5, max_iterations: 100 })
}).then(r => r.json());
```

### Render Word Cloud
```javascript
const wordCloudData = result.visualization_data.word_clouds[0].data;
// wordCloudData = [{text: "word", value: 0.025}, ...]
```

## ⚡ Performance Tips

1. **Start Small**: Use fewer iterations (50-80) for testing
2. **Vocabulary Control**: Adjust `min_df` and `max_df` to reduce vocab size
3. **Topic Count**: More topics = longer training
4. **Document Size**: Longer documents take more time to process

## ✅ Checklist for Production

- [ ] Test with your actual document corpus
- [ ] Tune parameters for your specific use case
- [ ] Validate coherence scores meet your threshold
- [ ] Test with expected document volume
- [ ] Implement caching for repeated queries
- [ ] Add monitoring for execution time
- [ ] Consider gensim for very large corpora (1000+ docs)

## 🎓 Learning Resources

### Understanding LDA
- Documents are mixtures of topics
- Topics are mixtures of words
- Algorithm discovers hidden topic structure
- Unsupervised - no labels needed

### Parameter Tuning Guide
- **Many small topics**: High n_topics, low alpha
- **Few broad topics**: Low n_topics, high alpha
- **Focused topics**: Low beta
- **Broad topics**: High beta

## 📞 Support

- Check test output: `python test_topic_modeling.py`
- Review logs: Check FastAPI console output
- Validate input: Min 5 documents, non-empty text
- Check response: `success` field indicates errors

---

**Ready to use!** Start with default parameters and adjust based on your corpus and requirements.
