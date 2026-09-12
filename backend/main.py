from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

from api.routes import ml, deep_learning, nlp, computer_vision, reinforcement_learning

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models and initialize resources
    logger.info("Starting up application...")
    logger.info("Loading machine learning models...")
    # TODO: Load and cache ML models here
    # Example:
    # app.state.models = {
    #     "classification_model": load_classification_model(),
    #     "nlp_model": load_nlp_model(),
    # }
    logger.info("Models loaded successfully")

    yield

    # Shutdown: Clean up resources
    logger.info("Shutting down application...")
    logger.info("Cleaning up resources...")
    # TODO: Clean up model resources if needed
    logger.info("Shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="AI Algorithms Demo API",
    description="Backend API for demonstrating various AI algorithms including ML, Deep Learning, NLP, Computer Vision, and Reinforcement Learning",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://frontend:3000",   # Docker frontend service
        *[o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()],
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "message": "AI Algorithms Demo API is running"
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to AI Algorithms Demo API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "ml": "/api/ml",
            "deep_learning": "/api/deep-learning",
            "nlp": "/api/nlp",
            "computer_vision": "/api/computer-vision",
            "reinforcement_learning": "/api/reinforcement-learning",
            "websocket": "/ws"
        }
    }


# WebSocket endpoint for real-time algorithm updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            logger.info(f"Received: {data}")

            # Echo back for now (will be replaced with actual algorithm processing)
            await websocket.send_text(f"Message received: {data}")

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")


# Include API routers
app.include_router(ml.router, prefix="/api")
app.include_router(deep_learning.router, prefix="/api")
app.include_router(nlp.router, prefix="/api")
app.include_router(computer_vision.router, prefix="/api")
app.include_router(reinforcement_learning.router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
