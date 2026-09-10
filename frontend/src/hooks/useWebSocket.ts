import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * WebSocket message structure
 */
interface WebSocketMessage {
  type: string;
  data?: any;
  message?: string;
  algorithm?: string;
}

/**
 * Return type for useWebSocket hook
 */
interface UseWebSocketReturn {
  sendMessage: (message: any) => void;
  lastMessage: WebSocketMessage | null;
  readyState: number;
  connect: () => void;
  disconnect: () => void;
}

/**
 * Options for configuring WebSocket behavior
 */
interface UseWebSocketOptions {
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (event: Event) => void;
  onMessage?: (message: WebSocketMessage) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
}

/**
 * Custom React hook for managing WebSocket connections
 *
 * @param url - The WebSocket URL to connect to
 * @param options - Configuration options for WebSocket behavior
 * @returns Object containing WebSocket utilities and state
 *
 * @example
 * ```typescript
 * const { sendMessage, lastMessage, readyState } = useWebSocket(
 *   'ws://localhost:8000/ws/client123',
 *   {
 *     onMessage: (msg) => console.log('Received:', msg),
 *     reconnect: true,
 *     reconnectInterval: 3000
 *   }
 * );
 * ```
 */
export const useWebSocket = (
  url: string,
  options?: UseWebSocketOptions
): UseWebSocketReturn => {
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [readyState, setReadyState] = useState<number>(WebSocket.CONNECTING);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>();

  /**
   * Establishes a WebSocket connection
   */
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setReadyState(WebSocket.OPEN);
      options?.onOpen?.();
      console.log('WebSocket connected');
    };

    ws.onclose = () => {
      setReadyState(WebSocket.CLOSED);
      options?.onClose?.();
      console.log('WebSocket closed');

      // Reconnect logic
      if (options?.reconnect) {
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect...');
          connect();
        }, options.reconnectInterval || 3000);
      }
    };

    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      options?.onError?.(event);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        setLastMessage(message);
        options?.onMessage?.(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };
  }, [url, options]);

  /**
   * Closes the WebSocket connection and clears reconnection attempts
   */
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    wsRef.current?.close();
  }, []);

  /**
   * Sends a message through the WebSocket connection
   *
   * @param message - The message to send (will be JSON stringified)
   */
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not open. Message not sent.');
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    sendMessage,
    lastMessage,
    readyState,
    connect,
    disconnect,
  };
};
