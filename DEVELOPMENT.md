# Development Guide

## Quick Start

### Using Docker (Recommended)

```bash
# Start all services
docker-compose up --build

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down

# Rebuild specific service
docker-compose up --build backend
```

### Local Development

#### Backend

```bash
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run server
python main.py
# OR with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure Conventions

### Backend

- **`/api/routes/`** - API endpoint definitions (thin controllers)
- **`/algorithms/`** - Core algorithm implementations (business logic)
- **`/utils/`** - Shared utilities (logging, validation, helpers)
- **`main.py`** - Application entry point and configuration

### Adding New Features

#### 1. Add a New Algorithm

```python
# backend/algorithms/ml/my_algorithm.py
from typing import Dict, Any
import numpy as np

class MyAlgorithm:
    def __init__(self):
        self.model = None
    
    def train(self, data: np.ndarray) -> Dict[str, Any]:
        """Train the algorithm."""
        # Implementation here
        return {"status": "trained"}
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Make predictions."""
        # Implementation here
        return predictions
```

#### 2. Add API Endpoint

```python
# backend/api/routes/ml.py
from algorithms.ml.my_algorithm import MyAlgorithm

@router.post("/my-algorithm/train")
async def train_my_algorithm(data: TrainingData):
    algorithm = MyAlgorithm()
    result = algorithm.train(data.values)
    return result
```

#### 3. Test the Endpoint

```bash
# Health check
curl http://localhost:8000/health

# Test endpoint
curl -X POST http://localhost:8000/api/ml/my-algorithm/train \
  -H "Content-Type: application/json" \
  -d '{"values": [1, 2, 3, 4, 5]}'
```

## API Testing

### Using Swagger UI

Visit http://localhost:8000/docs for interactive API documentation.

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Get ML algorithms
curl http://localhost:8000/api/ml

# WebSocket test (using websocat)
websocat ws://localhost:8000/ws
```

### Using Python

```python
import requests

# Test health endpoint
response = requests.get("http://localhost:8000/health")
print(response.json())

# Test ML endpoint
response = requests.get("http://localhost:8000/api/ml")
print(response.json())
```

## Common Tasks

### Install New Python Dependency

```bash
# Activate virtual environment
source venv/bin/activate

# Install package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt

# Rebuild Docker container
docker-compose up --build backend
```

### Database Migrations (When Added)

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Running Tests (When Added)

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_ml.py
```

## Debugging

### Backend Debugging

1. **Add breakpoints** using `import pdb; pdb.set_trace()`
2. **Check logs** in terminal or Docker logs
3. **Use FastAPI docs** at http://localhost:8000/docs

### Docker Debugging

```bash
# View logs
docker-compose logs -f backend

# Execute commands in container
docker-compose exec backend bash
docker-compose exec backend python -c "import torch; print(torch.__version__)"

# Check container status
docker-compose ps

# Inspect container
docker inspect ai-algorithms-backend
```

## Code Style

### Python

- Follow PEP 8 style guide
- Use type hints for function parameters and return values
- Add docstrings to all public functions and classes
- Maximum line length: 88 characters (Black formatter)

```python
def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the accuracy of predictions.
    
    Args:
        predictions: Array of predicted values
        labels: Array of true labels
    
    Returns:
        Accuracy score as a float between 0 and 1
    """
    return np.mean(predictions == labels)
```

### API Design

- Use RESTful conventions
- Version APIs when making breaking changes
- Return consistent error formats
- Use appropriate HTTP status codes

## Performance Optimization

### Model Loading

Models should be loaded during app startup in the lifespan context:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup
    app.state.models = {
        "classifier": load_model("classifier.pkl"),
    }
    yield
    # Cleanup
```

### Caching

Use Python's `functools.lru_cache` for expensive operations:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def preprocess_data(data_hash: str):
    # Expensive preprocessing
    return processed_data
```

## Environment Variables

Create a `.env` file in the backend directory:

```bash
# .env
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO

# API Keys (example)
OPENAI_API_KEY=your_key_here
```

Load in code:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    environment: str = "development"
    debug: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure virtual environment is activated
2. **Port conflicts**: Change port in docker-compose.yml or locally
3. **NLTK data errors**: Run NLTK download commands
4. **Docker build fails**: Clear cache with `docker system prune -a`

### Getting Help

- Check the README.md for basic setup
- Review API documentation at /docs
- Check Docker logs for error messages
- Verify all dependencies are installed

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-algorithm

# Make changes and commit
git add .
git commit -m "Add new ML algorithm"

# Push to remote
git push origin feature/my-algorithm

# Create pull request on GitHub
```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Docker Documentation](https://docs.docker.com/)
