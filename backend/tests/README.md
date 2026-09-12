# Backend Tests

This directory contains the test suite for the AI Algorithms Demo backend API.

## Test Structure

- `conftest.py` - Pytest fixtures and configuration
- `test_api.py` - Tests for API endpoints
- `test_datasets.py` - Tests for dataset utilities

## Running Tests

### Using Docker (Recommended)

```bash
# From the backend directory
docker build -t ai-algorithms-backend .
docker run --rm ai-algorithms-backend pytest

# Or run specific test files
docker run --rm ai-algorithms-backend pytest tests/test_datasets.py -v

# Run with coverage report
docker run --rm ai-algorithms-backend pytest --cov=backend --cov-report=term-missing
```

### Using Virtual Environment (Local Development)

```bash
# Create virtual environment (first time only)
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run specific test file
pytest tests/test_datasets.py -v

# Run with coverage
pytest --cov=backend --cov-report=html

# Run tests matching a pattern
pytest -k "test_get_iris"
```

### Using Docker Compose

If the project has docker-compose configuration:

```bash
# From the project root
docker-compose run backend pytest
```

## Test Configuration

Test configuration is defined in `pytest.ini`:

- Test discovery: Finds all files matching `test_*.py`
- Coverage reporting: Generates both terminal and HTML reports
- Verbose output: Shows detailed test results

## Writing New Tests

When adding new tests:

1. Create test files with the `test_` prefix
2. Use fixtures from `conftest.py` for common setup
3. Follow the existing naming convention: `test_<functionality>`
4. Add docstrings to describe what each test validates

Example:

```python
def test_new_feature(client: TestClient):
    """Test description here"""
    response = client.get("/api/endpoint")
    assert response.status_code == 200
```

## Current Test Coverage

- **API Tests**: Health check, root endpoint, basic structure
- **Dataset Tests**: All dataset loaders, normalization, validation, error handling

## TODO

- [ ] Add tests for ML algorithm endpoints as they are implemented
- [ ] Add tests for WebSocket functionality
- [ ] Add integration tests for end-to-end workflows
- [ ] Add performance/load tests
