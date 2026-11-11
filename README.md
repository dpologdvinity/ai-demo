# AI Algorithms Demonstration Website

A comprehensive web application for demonstrating various AI algorithms across multiple domains including Machine Learning, Deep Learning, Natural Language Processing, Computer Vision, Reinforcement Learning, and Classical AI (search/optimization).

## Project Structure

```
ai-demo/
├── backend/                          # Python FastAPI backend
│   ├── api/                         # API layer
│   │   ├── __init__.py
│   │   └── routes/                  # API route handlers
│   │       ├── __init__.py
│   │       ├── ml.py                # Machine Learning endpoints
│   │       ├── deep_learning.py     # Deep Learning endpoints
│   │       ├── nlp.py               # NLP endpoints
│   │       ├── computer_vision.py   # Computer Vision endpoints
│   │       ├── reinforcement_learning.py  # RL endpoints
│   │       └── classical_ai.py      # Search/optimization endpoints
│   ├── algorithms/                  # Algorithm implementations
│   │   ├── __init__.py
│   │   ├── ml/                      # Classical ML algorithms
│   │   │   └── __init__.py
│   │   ├── deep_learning/           # Neural network algorithms
│   │   │   └── __init__.py
│   │   ├── nlp/                     # NLP algorithms
│   │   │   └── __init__.py
│   │   ├── computer_vision/         # CV algorithms
│   │   │   └── __init__.py
│   │   ├── reinforcement_learning/  # RL algorithms
│   │   │   └── __init__.py
│   │   └── classical_ai/            # Search/optimization algorithms
│   │       └── __init__.py
│   ├── utils/                       # Utility functions
│   │   └── __init__.py
│   ├── requirements.txt             # Python dependencies
│   ├── Dockerfile                   # Backend Docker configuration
│   └── main.py                      # FastAPI application entry point
├── frontend/                         # React 18 + TypeScript frontend
├── .claude/settings.json            # Shared Claude Code project config
├── docker-compose.yml               # Docker Compose configuration
├── AGENTS.md                        # Instructions for AI coding agents
├── CLAUDE.md                        # Imports AGENTS.md for Claude Code
├── CONTRIBUTING.md                  # Setup, checks, commit conventions
└── README.md                        # This file
```

## Technology Stack

### Backend

- **Framework**: FastAPI 0.115.0
- **Server**: Uvicorn with WebSocket support
- **Machine Learning**: scikit-learn, PyTorch, TensorFlow
- **NLP**: NLTK, spaCy
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Plotly
- **Security**: python-jose, passlib

### Frontend
- **Framework**: React 18
- **Language**: TypeScript
- **Build Tool**: Vite
- **Routing**: React Router
- **Server State**: TanStack Query
- **Styling**: Tailwind CSS
- **UI Components**: Shadcn/UI
- **Visualization**: Recharts
- **HTTP Client**: Axios
- **Icons**: Lucide React

## Features

### Current Implementation

- RESTful API with FastAPI
- WebSocket support for real-time algorithm updates
- CORS middleware configured for frontend integration
- Health check endpoint
- Modular route structure for different AI domains
- Lifespan events for model loading/cleanup
- Comprehensive API documentation (Swagger/ReDoc)

### Implemented Frontend Features
- Interactive algorithm demonstrations across multiple domains
- Real-time visualization of algorithm execution using Recharts
- Parameter tuning with responsive UI updates
- Educational explanations and algorithm descriptions
- Domain-based navigation (ML, Deep Learning, NLP, Computer Vision, RL)

### Planned Features

- Interactive algorithm demonstrations
- Real-time visualization of algorithm execution
- Educational explanations and parameter tuning
- Comparison tools for different algorithms
- Performance metrics and benchmarking

## Getting Started

### Prerequisites

- Docker and Docker Compose (recommended)
- OR Python 3.11+ and Node.js 20+ (for local development)

### Running with Docker (Recommended)

1. **Clone the repository**

   ```bash
   cd ~/ai-demo
   ```

2. **Build and start the services**

   ```bash
   docker-compose up --build
   ```

3. **Access the application**
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc
   - Frontend: http://localhost:3000

4. **Stop the services**
   ```bash
   docker-compose down
   ```

### Running Locally (Without Docker)

#### Backend Setup

1. **Navigate to backend directory**

   ```bash
   cd backend
   ```

2. **Create a virtual environment with [uv](https://docs.astral.sh/uv/)**

   ```bash
   uv venv
   ```

3. **Install dependencies** (CPU-only PyTorch; no GPU needed)

   ```bash
   uv pip install --torch-backend cpu -r requirements.txt
   ```

   The NLP tokenizer downloads its NLTK data on first use, so you don't need to download it by hand.

4. **Run the server**

   ```bash
   uv run uvicorn main:app --reload --port 8000
   ```

5. **Access the API**
   - API Root: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## API Endpoints

### Core Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check status
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)
- `WS /ws` - WebSocket connection for real-time updates

### Algorithm Endpoints

- `GET /api/ml` - Machine Learning algorithms
- `GET /api/deep-learning` - Deep Learning algorithms
- `GET /api/nlp` - Natural Language Processing algorithms
- `GET /api/computer-vision` - Computer Vision algorithms
- `GET /api/reinforcement-learning` - Reinforcement Learning algorithms
- `GET /api/classical-ai` - Classical AI search/optimization algorithms

## Deployment

The backend deploys to Render (`render.yaml`, Docker-based) and the
frontend to Vercel (`frontend/vercel.json`).

1. **Backend (Render)**: create a service from `render.yaml`. Set the
   `CORS_ORIGINS` env var to the frontend's Vercel URL (e.g.
   `https://your-app.vercel.app`) so the browser is allowed to call the
   API cross-origin.
2. **Frontend (Vercel)**: set the `VITE_API_URL` env var in the Vercel
   project settings to the Render backend's URL (e.g.
   `https://ai-demo-backend.onrender.com`). This is required — Vercel
   has no reverse proxy like nginx/Vite dev server, so without it API
   calls go to the frontend's own origin and fail.

## Development

### Adding New Algorithms

1. **Implement the algorithm** in the appropriate `backend/algorithms/` subdirectory
2. **Create API endpoint** in the corresponding `backend/api/routes/` file
3. **Update documentation** in this README

### Continuous Integration

`.github/workflows/ci.yml` runs on pushes and pull requests to `master` (and manually).
It only verifies; Render deploys from `render.yaml` and CI never deploys. No secrets are needed.
Jobs and the local commands that mirror them:

```bash
# backend (Python 3.11): pytest with coverage
cd backend && uv venv && uv pip install --torch-backend cpu -r requirements.txt pytest-cov
uv run --no-sync pytest --cov=. --cov-report=term

# frontend: lint and build
cd frontend && npm ci && npm run lint && npm run build

# docker: build only, no push
docker build --tag ai-demo-backend:ci backend
```

### Project Guidelines

- Follow PEP 8 style guide for Python code
- Use type hints for function parameters and return values
- Add docstrings to all functions and classes
- Write unit tests for algorithm implementations
- Keep routes thin - business logic belongs in algorithm modules

## Environment Variables

The following environment variables can be configured:

- `ENVIRONMENT` - Set to `development` or `production` (default: `development`)
- `PYTHONUNBUFFERED` - Set to `1` for unbuffered Python output
- `NODE_ENV` - Frontend environment (development or production)
- `VITE_API_URL` - Frontend API base URL. Empty by default, which routes
  requests through the same-origin dev proxy (`vite.config.ts`) or the
  nginx `/api` proxy in Docker. Set this explicitly only when the frontend
  and backend are deployed to different origins (e.g. Vercel + Render —
  see Deployment below).
- `CORS_ORIGINS` - Comma-separated extra origins the backend should accept
  requests from, appended to the built-in localhost/Docker defaults
  (used on Render; see `render.yaml`).

## Troubleshooting

### Backend Issues

**Import errors**: Make sure all dependencies are installed, and run commands through `uv run` from `backend/`

```bash
uv pip install --torch-backend cpu -r requirements.txt
```

**Port conflicts**: Change the port in `docker-compose.yml` or when running locally

```bash
uvicorn main:app --reload --port 8001
```

**NLTK data errors**: The tokenizer downloads `punkt_tab` and `stopwords` on first use. Offline machines need them fetched in advance

```bash
uv run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### Docker Issues

**Build failures**: Clear Docker cache and rebuild

```bash
docker-compose down
docker system prune -a
docker-compose up --build
```

**Volume permissions**: Ensure proper file permissions for mounted volumes

## Next Steps

1. Expand algorithm implementations for each domain
2. Advanced interactive visualizations and analytics
3. User authentication and session management
4. Database integration for saving user progress
5. Performance optimization and caching strategies

## License

[MIT](LICENSE)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the CI-equivalent checks and the commit conventions.

## Working with AI Agents

This repo is set up for AI coding agents, and the setup is public as an example of the workflow:

- [AGENTS.md](AGENTS.md) is the single source of truth for agent instructions: commands, architecture and conventions. [CLAUDE.md](CLAUDE.md) is a one-line `@AGENTS.md` import, so Claude Code and Codex read the same file.
- [.claude/settings.json](.claude/settings.json) holds the shared Claude Code project config (enabled plugins). Personal settings stay in the gitignored `.claude/settings.local.json`.
- `docs/agents/specs/` and `docs/agents/plans/` are where design specs and implementation plans from agent sessions are kept.

## Contact

For questions or support, please open an issue in the repository.
