# NER API Reference

## Quick Start

### Base URL
```
/api/nlp/ner
```

## Endpoints

### 1. Extract Entities

**Endpoint**: `POST /api/nlp/ner/extract`

**Description**: Extract named entities from text using spaCy NER model.

**Request Body**:
```json
{
  "model_name": "en_core_web_sm",          // Optional: spaCy model to use
  "entity_types": [                        // Optional: Entity types to extract
    "PERSON", "ORG", "GPE", "DATE"
  ],
  "confidence_threshold": 0.0,             // Optional: Min confidence (0.0-1.0)
  "text_index": 0,                         // Optional: Sample text index (0-29)
  "merge_entities": true,                  // Optional: Merge adjacent entities
  "use_custom_text": false,                // Optional: Use custom text
  "custom_text": null                      // Optional: Custom text string
}
```

**Response**:
```json
{
  "success": true,
  "metrics": {
    "total_entities": 15,
    "entity_types": 4,
    "text_length": 235,
    "entities_per_100_chars": 6.38
  },
  "entities": [
    {
      "text": "Apple",
      "label": "ORG",
      "start": 0,
      "end": 5,
      "confidence": 1.0
    }
  ],
  "entity_distribution": {
    "ORG": 3,
    "PERSON": 2,
    "GPE": 5,
    "DATE": 5
  },
  "annotated_text": [
    {
      "text": "Apple",
      "label": "ORG",
      "start": 0,
      "end": 5
    },
    {
      "text": " CEO ",
      "label": null,
      "start": 5,
      "end": 10
    }
  ],
  "visualization_data": {
    "text_title": "Tech Industry News",
    "total_entities": 15,
    "entity_types_found": 4,
    "entity_type_info": {
      "PERSON": {
        "description": "People, including fictional characters",
        "color": "#3b82f6",
        "examples": "Tim Cook, Jennifer Doudna"
      }
    },
    "distribution_chart": [
      {
        "type": "GPE",
        "count": 5,
        "color": "#f59e0b",
        "percentage": 33.3
      }
    ],
    "top_entities": [
      {
        "text": "Apple",
        "count": 2,
        "label": "ORG"
      }
    ]
  },
  "execution_time_ms": 45.2,
  "parameters_used": {
    "model_name": "en_core_web_sm",
    "entity_types": ["PERSON", "ORG", "GPE", "DATE"],
    "confidence_threshold": 0.0,
    "text_index": 0,
    "merge_entities": true
  },
  "model_info": {
    "model_name": "en_core_web_sm",
    "spacy_version": "3.7.6",
    "supports_confidence": false
  }
}
```

---

### 2. Get NER Info

**Endpoint**: `GET /api/nlp/ner/info`

**Description**: Get complete NER algorithm information including metadata, parameters, entity types, and sample texts.

**Response**:
```json
{
  "metadata": {
    "id": "ner",
    "name": "Named Entity Recognition (NER)",
    "slug": "ner",
    "category": "nlp",
    "description": "Identify and classify named entities in text",
    "difficulty": "intermediate",
    "tags": ["nlp", "ner", "entity-extraction"],
    "use_cases": [
      "Information extraction",
      "Document indexing",
      "Knowledge graph construction"
    ],
    "complexity": {
      "time": "O(n)",
      "space": "O(entities)"
    },
    "parameters": [...],
    "theory": "...",
    "pros": [...],
    "cons": [...],
    "related_algorithms": [...]
  },
  "entity_types": {
    "PERSON": {
      "description": "People, including fictional characters",
      "color": "#3b82f6",
      "examples": "Tim Cook, Jennifer Doudna"
    }
  },
  "sample_texts": [
    {
      "id": "tech_news",
      "title": "Tech Industry News",
      "text": "Apple CEO Tim Cook announced..."
    }
  ],
  "total_samples": 32,
  "model_info": {
    "default_model": "en_core_web_sm",
    "supported_models": ["en_core_web_sm", "en_core_web_md"],
    "requires_download": true,
    "library": "spaCy (en_core_web_sm model)",
    "entity_types_supported": [
      "PERSON", "ORG", "GPE", "DATE", "TIME",
      "MONEY", "PERCENT", "PRODUCT", "EVENT",
      "LOC", "FAC", "NORP", "WORK_OF_ART",
      "LAW", "LANGUAGE"
    ]
  }
}
```

---

### 3. Get Sample Texts

**Endpoint**: `GET /api/nlp/ner/samples`

**Description**: Get all sample texts with entity type information.

**Response**:
```json
{
  "samples": [
    {
      "id": "tech_news",
      "title": "Tech Industry News",
      "text": "Apple CEO Tim Cook announced..."
    },
    {
      "id": "business",
      "title": "Business Announcement",
      "text": "Microsoft Corporation completed..."
    }
  ],
  "entity_types": {
    "PERSON": {
      "description": "People, including fictional characters",
      "color": "#3b82f6",
      "examples": "Tim Cook, Jennifer Doudna"
    }
  },
  "total_samples": 32
}
```

---

## Entity Types

| Type | Color | Description | Examples |
|------|-------|-------------|----------|
| PERSON | #3b82f6 (blue) | People, including fictional characters | Tim Cook, Jennifer Doudna |
| ORG | #10b981 (green) | Companies, agencies, institutions | Apple, Microsoft, UN |
| GPE | #f59e0b (orange) | Geopolitical entities | California, Paris, USA |
| DATE | #8b5cf6 (purple) | Dates or periods | June 8, 2023, Thursday |
| TIME | #ec4899 (pink) | Times smaller than a day | 3:00 PM, morning |
| MONEY | #14b8a6 (teal) | Monetary values | $69 billion, 100 euros |
| PERCENT | #f97316 (orange-red) | Percentage values | 50%, three percent |
| PRODUCT | #06b6d4 (cyan) | Objects, vehicles, foods | iPhone 15, Cybertruck |
| EVENT | #84cc16 (lime) | Named events | World Cup, COP28 |
| LOC | #eab308 (yellow) | Non-GPE locations | Lusail Stadium |
| FAC | #64748b (slate) | Buildings, airports | Gigafactory, Museum |
| NORP | #a855f7 (purple-alt) | Nationalities, religions | American, Democratic |
| WORK_OF_ART | #ec4899 (pink-alt) | Titles of creative works | Oppenheimer |
| LAW | #78716c (stone) | Named laws | Constitution |
| LANGUAGE | #06b6d4 (cyan-alt) | Named languages | English, Spanish |

---

## Usage Examples

### Example 1: Extract Entities from Sample Text

```bash
curl -X POST http://localhost:8000/api/nlp/ner/extract \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "en_core_web_sm",
    "entity_types": ["PERSON", "ORG", "GPE"],
    "text_index": 0,
    "merge_entities": true
  }'
```

### Example 2: Extract Entities from Custom Text

```bash
curl -X POST http://localhost:8000/api/nlp/ner/extract \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "en_core_web_sm",
    "entity_types": ["PERSON", "ORG", "DATE"],
    "use_custom_text": true,
    "custom_text": "Elon Musk founded SpaceX in 2002 in Los Angeles, California."
  }'
```

### Example 3: Get All Sample Texts

```bash
curl http://localhost:8000/api/nlp/ner/samples
```

### Example 4: Get Algorithm Information

```bash
curl http://localhost:8000/api/nlp/ner/info
```

---

## JavaScript/TypeScript Examples

### Fetch Sample Texts

```typescript
const response = await fetch('/api/nlp/ner/samples');
const { samples, entity_types, total_samples } = await response.json();

console.log(`Found ${total_samples} sample texts`);
samples.forEach(sample => {
  console.log(`${sample.id}: ${sample.title}`);
});
```

### Extract Entities

```typescript
interface NERRequest {
  model_name?: string;
  entity_types?: string[];
  confidence_threshold?: number;
  text_index?: number;
  merge_entities?: boolean;
  use_custom_text?: boolean;
  custom_text?: string;
}

async function extractEntities(params: NERRequest) {
  const response = await fetch('/api/nlp/ner/extract', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

// Usage
const result = await extractEntities({
  model_name: 'en_core_web_sm',
  entity_types: ['PERSON', 'ORG', 'GPE'],
  text_index: 0,
  merge_entities: true
});

console.log(`Found ${result.metrics.total_entities} entities`);
result.entities.forEach(entity => {
  console.log(`${entity.text} (${entity.label})`);
});
```

### Render Highlighted Text

```typescript
function renderHighlightedText(annotatedSpans, entityTypeInfo) {
  return annotatedSpans.map(span => {
    if (span.label) {
      const color = entityTypeInfo[span.label].color;
      return `<mark style="background-color: ${color}20; border-bottom: 2px solid ${color}">
        ${span.text}
        <span class="entity-label">${span.label}</span>
      </mark>`;
    } else {
      return span.text;
    }
  }).join('');
}

// Usage
const result = await extractEntities({ text_index: 0 });
const html = renderHighlightedText(
  result.annotated_text,
  result.visualization_data.entity_type_info
);
document.getElementById('ner-output').innerHTML = html;
```

---

## React Component Example

```tsx
import React, { useState, useEffect } from 'react';

interface Entity {
  text: string;
  label: string;
  start: number;
  end: number;
  confidence: number;
}

interface NERResult {
  entities: Entity[];
  annotated_text: Array<{
    text: string;
    label: string | null;
    start: number;
    end: number;
  }>;
  visualization_data: {
    entity_type_info: Record<string, {
      description: string;
      color: string;
      examples: string;
    }>;
    distribution_chart: Array<{
      type: string;
      count: number;
      color: string;
      percentage: number;
    }>;
  };
}

export function NERDemo() {
  const [result, setResult] = useState<NERResult | null>(null);
  const [selectedTypes, setSelectedTypes] = useState(['PERSON', 'ORG', 'GPE']);
  const [textIndex, setTextIndex] = useState(0);

  const extractEntities = async () => {
    const response = await fetch('/api/nlp/ner/extract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        entity_types: selectedTypes,
        text_index: textIndex,
        merge_entities: true
      })
    });
    const data = await response.json();
    setResult(data);
  };

  useEffect(() => {
    extractEntities();
  }, [selectedTypes, textIndex]);

  if (!result) return <div>Loading...</div>;

  return (
    <div>
      <h2>Named Entity Recognition</h2>
      
      {/* Highlighted Text */}
      <div className="ner-text">
        {result.annotated_text.map((span, i) => (
          <span
            key={i}
            style={{
              backgroundColor: span.label 
                ? result.visualization_data.entity_type_info[span.label].color + '20'
                : 'transparent',
              borderBottom: span.label 
                ? `2px solid ${result.visualization_data.entity_type_info[span.label].color}`
                : 'none'
            }}
          >
            {span.text}
          </span>
        ))}
      </div>

      {/* Entity List */}
      <div className="entity-list">
        <h3>Extracted Entities ({result.entities.length})</h3>
        <ul>
          {result.entities.map((entity, i) => (
            <li key={i}>
              <strong>{entity.text}</strong> ({entity.label})
            </li>
          ))}
        </ul>
      </div>

      {/* Distribution Chart */}
      <div className="distribution">
        <h3>Entity Distribution</h3>
        {result.visualization_data.distribution_chart.map(item => (
          <div key={item.type} style={{ display: 'flex', alignItems: 'center' }}>
            <div 
              style={{ 
                width: 20, 
                height: 20, 
                backgroundColor: item.color,
                marginRight: 10
              }}
            />
            <span>{item.type}: {item.count} ({item.percentage}%)</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

| Status Code | Error | Description |
|-------------|-------|-------------|
| 400 | Invalid parameters | Parameters validation failed |
| 404 | Not found | Resource not found |
| 500 | Server error | Internal server error during processing |

### Example Error Handling

```typescript
try {
  const result = await extractEntities({
    text_index: 100  // Invalid - out of range
  });
} catch (error) {
  if (error.status === 400) {
    console.error('Invalid parameters:', error.detail);
  } else if (error.status === 500) {
    console.error('Server error:', error.detail);
  }
}
```

---

## Notes

1. **Model Download Required**: Before using NER, ensure the spaCy model is downloaded:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Performance**: First request may take ~500ms to load the model. Subsequent requests are much faster (20-100ms).

3. **Entity Types**: You can extract all types by omitting the `entity_types` parameter or passing an empty array.

4. **Custom Text**: When using custom text, the `text_index` parameter is ignored.

5. **Confidence Scores**: The default spaCy models don't provide confidence scores, so all entities have confidence=1.0. For models with confidence, adjust the `confidence_threshold` parameter.

---

**API Version**: 1.0  
**Last Updated**: August 7, 2026  
**Status**: Production Ready ✅
