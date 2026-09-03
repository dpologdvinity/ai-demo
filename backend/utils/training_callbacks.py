"""Training callbacks for streaming real-time progress updates.

This module provides callback classes for capturing and streaming training
progress from various machine learning algorithms, including both generic
neural network training and sklearn iterative algorithms.
"""

import asyncio
from typing import AsyncGenerator, Dict, Any, List
import time


class TrainingCallback:
    """Base class for streaming training progress.

    This callback can be used with training loops to capture metrics at the
    end of each epoch and stream them asynchronously.

    Attributes:
        metrics: List of dictionaries containing epoch metrics.
    """

    def __init__(self) -> None:
        """Initialize the callback with an empty metrics list."""
        self.metrics: List[Dict[str, Any]] = []

    async def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called at end of each training epoch.

        Args:
            epoch: The epoch number (0-indexed).
            metrics: Dictionary containing metric names and values for this epoch.
        """
        self.metrics.append({'epoch': epoch, **metrics})
        await asyncio.sleep(0)  # Yield control to event loop

    async def stream_metrics(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream accumulated metrics as an async generator.

        Yields:
            Dictionary containing epoch number and associated metrics.
        """
        for metric in self.metrics:
            yield metric
            await asyncio.sleep(0.1)  # Small delay for visualization


class SklearnProgressCallback:
    """Callback for sklearn iterative algorithms.

    This callback can be passed to sklearn's iterative algorithms (e.g.,
    SGDClassifier, MLPClassifier) to capture iteration-level progress.

    Attributes:
        iterations: List of dictionaries containing iteration metrics.
    """

    def __init__(self) -> None:
        """Initialize the callback with an empty iterations list."""
        self.iterations: List[Dict[str, Any]] = []

    def __call__(self, iteration: int, loss: float) -> None:
        """Called at each iteration of the sklearn algorithm.

        Args:
            iteration: The current iteration number.
            loss: The loss value at this iteration.
        """
        self.iterations.append({
            'iteration': iteration,
            'loss': loss,
            'timestamp': time.time()
        })

    async def stream_iterations(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream accumulated iteration metrics as an async generator.

        Yields:
            Dictionary containing iteration number, loss, and timestamp.
        """
        for iteration_data in self.iterations:
            yield iteration_data
            await asyncio.sleep(0.1)  # Small delay for visualization
