# BERT Fine-tuning Frontend Integration Guide

## Quick Start

### API Endpoints

#### 1. Get Algorithm Info
```javascript
GET /nlp/bert-finetuning/info

// Response
{
  "metadata": {
    "id": "bert-finetuning",
    "name": "BERT Fine-tuning",
    "slug": "bert-finetuning",
    "category": "nlp",
    "description": "Fine-tune pre-trained BERT for text classification",
    "difficulty": "advanced",
    "tags": ["nlp", "transformers", "transfer-learning", "text-classification", "deep-learning"],
    "parameters": [...],
    "complexity": {
      "time": "O(T²×d×L) where T=sequence length, d=hidden dim, L=layers",
      "space": "O(model_params)"
    }
  },
  "dataset": {
    "name": "Sentiment Classification Dataset",
    "total_samples": 45,
    "num_classes": 3,
    "class_names": ["Negative", "Neutral", "Positive"]
  },
  "model_info": {
    "library": "Hugging Face Transformers",
    "supported_models": ["bert-base-uncased", "distilbert-base-uncased"],
    "parameters_bert_base": "110M",
    "parameters_distilbert": "66M"
  }
}
```

#### 2. Train Model
```javascript
POST /nlp/bert-finetuning/train

// Request Body
{
  "model_name": "bert-base-uncased",  // or "distilbert-base-uncased"
  "learning_rate": 0.00002,           // 2e-5 (default)
  "epochs": 3,                         // 1-10
  "batch_size": 16,                    // 8, 16, 24, 32
  "max_length": 128,                   // 64, 128, 256, 512
  "use_custom_dataset": false,         // optional
  "custom_texts": [],                  // optional: list of strings
  "custom_labels": []                  // optional: list of 0/1/2
}

// Response - see detailed structure below
```

## Response Structure

### Training Response
```typescript
interface BERTFinetuningResponse {
  success: boolean;
  training_history: TrainingHistory[];
  predictions: PredictionResult[];
  confusion_matrix: ConfusionMatrix;
  attention_visualizations: AttentionVisualization[];
  metrics: Metrics;
  visualization_data: VisualizationData;
  execution_time_ms: number;
  parameters_used: object;
  model_info: ModelInfo;
  error?: string;
}

interface TrainingHistory {
  epoch: number;
  train_loss: number;
  train_accuracy: number;
  val_loss: number;
  val_accuracy: number;
}

interface PredictionResult {
  text: string;
  predicted_label: number;        // 0, 1, or 2
  predicted_class: string;        // "Negative", "Neutral", "Positive"
  confidence: number;             // 0.0 to 1.0
  probabilities: number[];        // [prob_negative, prob_neutral, prob_positive]
  true_label?: number;
}

interface ConfusionMatrix {
  matrix: number[][];             // 3x3 matrix
  labels: string[];               // ["Negative", "Neutral", "Positive"]
  accuracy: number;
  precision: number[];            // per class
  recall: number[];               // per class
  f1_score: number[];             // per class
}

interface AttentionVisualization {
  text: string;
  tokens: string[];               // tokenized words including [CLS], [SEP]
  attention_weights: number[][];  // NxN matrix where N = num_tokens
  sample_index: number;
}

interface Metrics {
  final_train_loss: number;
  final_train_accuracy: number;
  final_val_loss: number;
  final_val_accuracy: number;
  final_accuracy: number;
  average_confidence: number;
  total_epochs: number;
  num_predictions: number;
  num_classes: number;
}

interface ModelInfo {
  model_name: string;
  num_parameters: number;
  num_trainable_parameters: number;
  num_layers: number;
  hidden_size: number;
  num_attention_heads: number;
  vocab_size: number;
  max_position_embeddings: number;
  num_classes: number;
  class_names: string[];
  device: string;                 // "cuda:0" or "cpu"
}
```

## React Component Examples

### 1. Parameter Controls Component

```jsx
import React, { useState } from 'react';

function BERTParameterControls({ onTrain }) {
  const [params, setParams] = useState({
    model_name: 'bert-base-uncased',
    learning_rate: 0.00002,
    epochs: 3,
    batch_size: 16,
    max_length: 128
  });

  const handleTrain = async () => {
    const response = await fetch('/nlp/bert-finetuning/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    const data = await response.json();
    onTrain(data);
  };

  return (
    <div className="parameter-controls">
      <div className="form-group">
        <label>Model Variant</label>
        <select 
          value={params.model_name}
          onChange={(e) => setParams({...params, model_name: e.target.value})}
        >
          <option value="bert-base-uncased">BERT Base (110M params)</option>
          <option value="distilbert-base-uncased">DistilBERT (66M params)</option>
        </select>
      </div>

      <div className="form-group">
        <label>Learning Rate: {params.learning_rate.toExponential(1)}</label>
        <input
          type="range"
          min={1e-5}
          max={5e-5}
          step={5e-6}
          value={params.learning_rate}
          onChange={(e) => setParams({...params, learning_rate: parseFloat(e.target.value)})}
        />
      </div>

      <div className="form-group">
        <label>Epochs: {params.epochs}</label>
        <input
          type="range"
          min={1}
          max={10}
          step={1}
          value={params.epochs}
          onChange={(e) => setParams({...params, epochs: parseInt(e.target.value)})}
        />
      </div>

      <div className="form-group">
        <label>Batch Size: {params.batch_size}</label>
        <select 
          value={params.batch_size}
          onChange={(e) => setParams({...params, batch_size: parseInt(e.target.value)})}
        >
          <option value={8}>8</option>
          <option value={16}>16</option>
          <option value={24}>24</option>
          <option value={32}>32</option>
        </select>
      </div>

      <div className="form-group">
        <label>Max Sequence Length: {params.max_length}</label>
        <select 
          value={params.max_length}
          onChange={(e) => setParams({...params, max_length: parseInt(e.target.value)})}
        >
          <option value={64}>64</option>
          <option value={128}>128</option>
          <option value={256}>256</option>
          <option value={512}>512</option>
        </select>
      </div>

      <button onClick={handleTrain} className="btn-primary">
        Train Model
      </button>
    </div>
  );
}
```

### 2. Training Curves Component (Chart.js)

```jsx
import React from 'react';
import { Line } from 'react-chartjs-2';

function TrainingCurves({ trainingHistory }) {
  const data = {
    labels: trainingHistory.map(h => `Epoch ${h.epoch}`),
    datasets: [
      {
        label: 'Training Loss',
        data: trainingHistory.map(h => h.train_loss),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        yAxisID: 'y-loss',
      },
      {
        label: 'Validation Loss',
        data: trainingHistory.map(h => h.val_loss),
        borderColor: 'rgb(255, 159, 64)',
        backgroundColor: 'rgba(255, 159, 64, 0.1)',
        yAxisID: 'y-loss',
      },
      {
        label: 'Training Accuracy',
        data: trainingHistory.map(h => h.train_accuracy * 100),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.1)',
        yAxisID: 'y-accuracy',
      },
      {
        label: 'Validation Accuracy',
        data: trainingHistory.map(h => h.val_accuracy * 100),
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.1)',
        yAxisID: 'y-accuracy',
      }
    ]
  };

  const options = {
    responsive: true,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    scales: {
      'y-loss': {
        type: 'linear',
        position: 'left',
        title: {
          display: true,
          text: 'Loss'
        }
      },
      'y-accuracy': {
        type: 'linear',
        position: 'right',
        title: {
          display: true,
          text: 'Accuracy (%)'
        },
        min: 0,
        max: 100,
        grid: {
          drawOnChartArea: false,
        }
      }
    },
    plugins: {
      title: {
        display: true,
        text: 'Training Progress'
      }
    }
  };

  return (
    <div className="training-curves">
      <Line data={data} options={options} />
    </div>
  );
}
```

### 3. Predictions Table Component

```jsx
import React from 'react';

function PredictionsTable({ predictions }) {
  return (
    <div className="predictions-table">
      <h3>Sample Predictions</h3>
      <table>
        <thead>
          <tr>
            <th>Text</th>
            <th>Predicted</th>
            <th>Confidence</th>
            <th>Correct</th>
            <th>Probabilities</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((pred, idx) => (
            <tr key={idx} className={pred.predicted_label === pred.true_label ? 'correct' : 'incorrect'}>
              <td className="text-cell" title={pred.text}>
                {pred.text.length > 50 ? pred.text.slice(0, 50) + '...' : pred.text}
              </td>
              <td>
                <span className={`label label-${pred.predicted_class.toLowerCase()}`}>
                  {pred.predicted_class}
                </span>
              </td>
              <td>{(pred.confidence * 100).toFixed(1)}%</td>
              <td>
                {pred.true_label !== undefined && (
                  pred.predicted_label === pred.true_label ? '✓' : '✗'
                )}
              </td>
              <td>
                <div className="probability-bars">
                  {pred.probabilities.map((prob, i) => (
                    <div key={i} className="prob-bar">
                      <div 
                        className="prob-fill" 
                        style={{ width: `${prob * 100}%` }}
                        title={`${(prob * 100).toFixed(1)}%`}
                      />
                    </div>
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

### 4. Confusion Matrix Heatmap (D3.js or Chart.js)

```jsx
import React from 'react';

function ConfusionMatrixHeatmap({ confusionMatrix }) {
  const { matrix, labels, accuracy } = confusionMatrix;
  const maxValue = Math.max(...matrix.flat());

  const getCellColor = (value) => {
    const intensity = value / maxValue;
    return `rgba(54, 162, 235, ${intensity})`;
  };

  return (
    <div className="confusion-matrix">
      <h3>Confusion Matrix (Accuracy: {(accuracy * 100).toFixed(1)}%)</h3>
      <table className="heatmap">
        <thead>
          <tr>
            <th></th>
            {labels.map(label => (
              <th key={label}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th>{labels[i]}</th>
              {row.map((value, j) => (
                <td 
                  key={j}
                  style={{ 
                    backgroundColor: getCellColor(value),
                    color: value > maxValue / 2 ? 'white' : 'black'
                  }}
                >
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      
      <div className="metrics-summary">
        <h4>Per-Class Metrics</h4>
        <table>
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
            </tr>
          </thead>
          <tbody>
            {labels.map((label, i) => (
              <tr key={label}>
                <td>{label}</td>
                <td>{(confusionMatrix.precision[i] * 100).toFixed(1)}%</td>
                <td>{(confusionMatrix.recall[i] * 100).toFixed(1)}%</td>
                <td>{(confusionMatrix.f1_score[i] * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

### 5. Attention Visualization Heatmap

```jsx
import React from 'react';

function AttentionHeatmap({ attentionViz }) {
  const { text, tokens, attention_weights } = attentionViz;
  const maxWeight = Math.max(...attention_weights.flat());

  const getCellColor = (weight) => {
    const intensity = weight / maxWeight;
    // Blue color scale
    return `rgba(54, 162, 235, ${intensity})`;
  };

  return (
    <div className="attention-heatmap">
      <h4>Attention Weights: "{text}"</h4>
      <div className="heatmap-container">
        <table className="attention-matrix">
          <thead>
            <tr>
              <th></th>
              {tokens.map((token, i) => (
                <th key={i} className="token-header">
                  <div className="token-label">{token}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tokens.map((token, i) => (
              <tr key={i}>
                <th className="token-label">{token}</th>
                {attention_weights[i].map((weight, j) => (
                  <td 
                    key={j}
                    style={{ backgroundColor: getCellColor(weight) }}
                    title={`${token} → ${tokens[j]}: ${weight.toFixed(4)}`}
                  />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="attention-legend">
        <span>Low Attention</span>
        <div className="gradient-bar" />
        <span>High Attention</span>
      </div>
    </div>
  );
}
```

### 6. Model Info Card

```jsx
import React from 'react';

function ModelInfoCard({ modelInfo }) {
  return (
    <div className="model-info-card">
      <h3>Model Information</h3>
      <div className="info-grid">
        <div className="info-item">
          <label>Model:</label>
          <span>{modelInfo.model_name}</span>
        </div>
        <div className="info-item">
          <label>Parameters:</label>
          <span>{(modelInfo.num_parameters / 1e6).toFixed(1)}M</span>
        </div>
        <div className="info-item">
          <label>Layers:</label>
          <span>{modelInfo.num_layers}</span>
        </div>
        <div className="info-item">
          <label>Hidden Size:</label>
          <span>{modelInfo.hidden_size}</span>
        </div>
        <div className="info-item">
          <label>Attention Heads:</label>
          <span>{modelInfo.num_attention_heads}</span>
        </div>
        <div className="info-item">
          <label>Vocab Size:</label>
          <span>{modelInfo.vocab_size.toLocaleString()}</span>
        </div>
        <div className="info-item">
          <label>Device:</label>
          <span className={modelInfo.device.includes('cuda') ? 'device-gpu' : 'device-cpu'}>
            {modelInfo.device.includes('cuda') ? '🚀 GPU' : '💻 CPU'}
          </span>
        </div>
        <div className="info-item">
          <label>Classes:</label>
          <span>{modelInfo.class_names.join(', ')}</span>
        </div>
      </div>
    </div>
  );
}
```

### 7. Complete Page Component

```jsx
import React, { useState } from 'react';

function BERTFinetuningPage() {
  const [isTraining, setIsTraining] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleTrain = async (params) => {
    setIsTraining(true);
    setError(null);
    
    try {
      const response = await fetch('/nlp/bert-finetuning/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setResults(data);
      } else {
        setError(data.error || 'Training failed');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="bert-finetuning-page">
      <header>
        <h1>BERT Fine-tuning for Text Classification</h1>
        <p>Fine-tune pre-trained BERT models on sentiment classification</p>
      </header>

      <div className="content-grid">
        <aside className="sidebar">
          <BERTParameterControls onTrain={handleTrain} disabled={isTraining} />
        </aside>

        <main className="main-content">
          {isTraining && (
            <div className="loading-spinner">
              <div className="spinner" />
              <p>Training model... This may take a few minutes.</p>
            </div>
          )}

          {error && (
            <div className="error-message">
              <strong>Error:</strong> {error}
            </div>
          )}

          {results && (
            <>
              <section className="results-section">
                <TrainingCurves trainingHistory={results.training_history} />
              </section>

              <section className="results-section">
                <PredictionsTable predictions={results.predictions} />
              </section>

              <section className="results-section">
                <ConfusionMatrixHeatmap confusionMatrix={results.confusion_matrix} />
              </section>

              <section className="results-section">
                <h3>Attention Visualizations</h3>
                {results.attention_visualizations.map((viz, idx) => (
                  <AttentionHeatmap key={idx} attentionViz={viz} />
                ))}
              </section>

              <section className="results-section">
                <ModelInfoCard modelInfo={results.model_info} />
              </section>

              <section className="metrics-summary">
                <h3>Training Summary</h3>
                <div className="metrics-grid">
                  <div className="metric-card">
                    <label>Final Validation Accuracy</label>
                    <span className="metric-value">
                      {(results.metrics.final_val_accuracy * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="metric-card">
                    <label>Average Confidence</label>
                    <span className="metric-value">
                      {(results.metrics.average_confidence * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="metric-card">
                    <label>Training Time</label>
                    <span className="metric-value">
                      {(results.execution_time_ms / 1000).toFixed(1)}s
                    </span>
                  </div>
                </div>
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
```

## CSS Styling Examples

```css
.bert-finetuning-page {
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem;
}

.content-grid {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 2rem;
  margin-top: 2rem;
}

.parameter-controls {
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
}

.results-section {
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  margin-bottom: 2rem;
}

.predictions-table table {
  width: 100%;
  border-collapse: collapse;
}

.predictions-table th,
.predictions-table td {
  padding: 0.75rem;
  text-align: left;
  border-bottom: 1px solid #eee;
}

.predictions-table tr.correct {
  background-color: #f0fdf4;
}

.predictions-table tr.incorrect {
  background-color: #fef2f2;
}

.label {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.875rem;
  font-weight: 600;
}

.label-negative {
  background-color: #fee2e2;
  color: #991b1b;
}

.label-neutral {
  background-color: #e5e7eb;
  color: #374151;
}

.label-positive {
  background-color: #dcfce7;
  color: #166534;
}

.confusion-matrix .heatmap {
  margin: 1rem 0;
}

.confusion-matrix .heatmap td {
  width: 60px;
  height: 60px;
  text-align: center;
  font-weight: 600;
  border: 1px solid #ddd;
}

.attention-heatmap {
  margin-top: 1rem;
}

.attention-matrix td {
  width: 30px;
  height: 30px;
  border: 1px solid #eee;
}

.token-label {
  writing-mode: vertical-rl;
  text-orientation: mixed;
  padding: 0.5rem;
  font-size: 0.75rem;
}

.loading-spinner {
  text-align: center;
  padding: 3rem;
}

.spinner {
  border: 4px solid #f3f3f3;
  border-top: 4px solid #3b82f6;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.metric-card {
  background: #f9fafb;
  padding: 1rem;
  border-radius: 6px;
  text-align: center;
}

.metric-value {
  display: block;
  font-size: 2rem;
  font-weight: 700;
  color: #3b82f6;
  margin-top: 0.5rem;
}
```

## Testing the API

### Using curl
```bash
# Get info
curl http://localhost:8000/nlp/bert-finetuning/info | jq

# Train with default parameters
curl -X POST http://localhost:8000/nlp/bert-finetuning/train \
  -H "Content-Type: application/json" \
  -d '{"model_name": "bert-base-uncased", "epochs": 3}' | jq

# Train with custom parameters
curl -X POST http://localhost:8000/nlp/bert-finetuning/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "distilbert-base-uncased",
    "learning_rate": 0.00003,
    "epochs": 5,
    "batch_size": 16,
    "max_length": 128
  }' | jq
```

### Using fetch in browser console
```javascript
// Get info
fetch('/nlp/bert-finetuning/info')
  .then(r => r.json())
  .then(console.log);

// Train model
fetch('/nlp/bert-finetuning/train', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model_name: 'bert-base-uncased',
    epochs: 3
  })
})
  .then(r => r.json())
  .then(console.log);
```

## Performance Tips

1. **Loading States**: Always show loading indicators during training (can take 1-10 minutes)
2. **Progress Updates**: Consider WebSocket for real-time epoch progress (future enhancement)
3. **Error Handling**: Display user-friendly error messages
4. **Caching**: Cache model info response (doesn't change)
5. **Lazy Loading**: Load chart libraries only when needed
6. **Responsive Design**: Make tables scrollable on mobile

## Common Issues & Solutions

### Issue: Training takes too long
- **Solution**: Use DistilBERT (2x faster) or reduce epochs/batch_size

### Issue: Out of memory error
- **Solution**: Reduce batch_size or max_length

### Issue: Low accuracy
- **Solution**: Increase epochs, try different learning rate, or provide more training data

### Issue: Model download fails
- **Solution**: Check internet connection; models download automatically on first use

## Next Steps

1. Implement the React components
2. Add custom dataset upload feature
3. Add model saving/loading
4. Implement real-time training progress
5. Add export functionality for results
6. Create responsive mobile views

## Support

For questions or issues:
- Check backend logs for errors
- Verify API endpoints are accessible
- Test with curl first before implementing frontend
- Monitor browser console for JavaScript errors
