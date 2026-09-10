# AI Algorithms Demonstration Website

A comprehensive web application for demonstrating various AI algorithms across multiple domains including Machine Learning, Deep Learning, Natural Language Processing, Computer Vision, and Reinforcement Learning.

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
│   │       └── reinforcement_learning.py  # RL endpoints
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
│   │   └── reinforcement_learning/  # RL algorithms
│   │       └── __init__.py
│   ├── utils/                       # Utility functions
│   │   └── __init__.py
│   ├── requirements.txt             # Python dependencies
│   ├── Dockerfile                   # Backend Docker configuration
│   └── main.py                      # FastAPI application entry point
├── frontend/                         # React 18 + TypeScript frontend
├── docker-compose.yml               # Docker Compose configuration
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
- **State Management**: TanStack Query (server state), Zustand (client state)
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

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**

   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

5. **Run the server**

   ```bash
   python main.py
   # OR
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access the API**
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

## Development

### Adding New Algorithms

1. **Implement the algorithm** in the appropriate `backend/algorithms/` subdirectory
2. **Create API endpoint** in the corresponding `backend/api/routes/` file
3. **Update documentation** in this README

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
- `VITE_API_URL` - Frontend API URL configuration (default: http://localhost:8000)

## Troubleshooting

### Backend Issues

**Import errors**: Ensure virtual environment is activated and all dependencies are installed

```bash
pip install -r requirements.txt
```

**Port conflicts**: Change the port in `docker-compose.yml` or when running locally

```bash
uvicorn main:app --reload --port 8001
```

**NLTK data errors**: Download required NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
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
5. Deployment configuration for production
6. Performance optimization and caching strategies

## License

To be determined

## Contributing

Contributions are welcome! Please follow the development guidelines and submit pull requests for review.

## Contact

For questions or support, please open an issue in the repository.
