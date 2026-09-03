"""WebSocket connection manager for streaming real-time training progress.

This module provides a connection manager for handling WebSocket connections
and streaming algorithm training progress from backend to frontend.
"""

from typing import Dict, List, Any, Callable, AsyncGenerator
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio


class ConnectionManager:
    """Manages WebSocket connections for streaming algorithm training progress.

    This class handles multiple WebSocket connections, allowing for both
    personal messages to specific clients and broadcasting to all clients.
    It also provides functionality for streaming training progress updates
    in real-time.

    Attributes:
        active_connections: Dictionary mapping client IDs to lists of WebSocket connections.
    """

    def __init__(self) -> None:
        """Initialize the connection manager with an empty connections dictionary."""
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept and store websocket connection.

        Args:
            websocket: The WebSocket connection to accept.
            client_id: Unique identifier for the client.
        """
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)

    def disconnect(self, websocket: WebSocket, client_id: str) -> None:
        """Remove websocket connection.

        Args:
            websocket: The WebSocket connection to remove.
            client_id: Unique identifier for the client.
        """
        if client_id in self.active_connections:
            self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]

    async def send_personal_message(self, message: Dict[str, Any], client_id: str) -> None:
        """Send message to specific client.

        Args:
            message: Dictionary containing the message data to send.
            client_id: Unique identifier for the target client.
        """
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                await connection.send_json(message)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected clients.

        Args:
            message: Dictionary containing the message data to broadcast.
        """
        for connections in self.active_connections.values():
            for connection in connections:
                await connection.send_json(message)

    async def stream_training_progress(
        self,
        client_id: str,
        algorithm_name: str,
        callback_func: Callable[[], AsyncGenerator[Dict[str, Any], None]]
    ) -> None:
        """Stream training progress using a callback function.

        This method continuously streams training progress updates to a specific
        client by invoking the provided callback function and forwarding the
        yielded progress data.

        Args:
            client_id: Unique client identifier.
            algorithm_name: Name of algorithm being trained.
            callback_func: Async generator function that yields progress updates.
        """
        try:
            async for progress in callback_func():
                await self.send_personal_message({
                    'algorithm': algorithm_name,
                    'type': 'training_progress',
                    'data': progress
                }, client_id)
        except Exception as e:
            await self.send_personal_message({
                'type': 'error',
                'message': str(e)
            }, client_id)


# Global instance
manager: ConnectionManager = ConnectionManager()
