import { useState } from 'react';
import { useWebSocket } from './useWebSocket';

/**
 * Training progress data structure
 */
interface TrainingProgress {
  epoch?: number;
  iteration?: number;
  loss?: number;
  accuracy?: number;
  [key: string]: any;
}

/**
 * Return type for useTrainingStream hook
 */
interface UseTrainingStreamReturn {
  progress: TrainingProgress[];
  isTraining: boolean;
  error: string | null;
  startTraining: (algorithm: string, params: any) => void;
  stopTraining: () => void;
}

/**
 * Custom React hook for streaming training progress from backend
 *
 * This hook manages WebSocket communication for AI algorithm training,
 * tracking progress updates, errors, and training state.
 *
 * @param clientId - Unique identifier for this client's WebSocket connection
 * @returns Object containing training state and control functions
 *
 * @example
 * ```typescript
 * const { progress, isTraining, startTraining, stopTraining } = useTrainingStream('client-123');
 *
 * // Start training
 * startTraining('neural-network', {
 *   learningRate: 0.01,
 *   epochs: 100
 * });
 *
 * // Stop training
 * stopTraining();
 *
 * // Access progress
 * console.log(progress); // [{ epoch: 1, loss: 0.5 }, ...]
 * ```
 */
export const useTrainingStream = (
  clientId: string
): UseTrainingStreamReturn => {
  const [progress, setProgress] = useState<TrainingProgress[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const { sendMessage } = useWebSocket(
    `${wsProtocol}//${window.location.host}/ws/${clientId}`,
    {
      onMessage: (message) => {
        if (message.type === 'training_progress') {
          setProgress((prev) => [...prev, message.data]);
        } else if (message.type === 'training_complete') {
          setIsTraining(false);
        } else if (message.type === 'error') {
          setError(message.message || 'An error occurred');
          setIsTraining(false);
        }
      },
      reconnect: true,
      reconnectInterval: 3000,
    }
  );

  /**
   * Initiates training for a specified algorithm with given parameters
   *
   * @param algorithm - The algorithm identifier (e.g., 'neural-network', 'genetic-algorithm')
   * @param params - Algorithm-specific parameters (learning rate, population size, etc.)
   */
  const startTraining = (algorithm: string, params: any) => {
    setProgress([]);
    setError(null);
    setIsTraining(true);
    sendMessage({
      type: 'start_training',
      algorithm,
      params,
    });
  };

  /**
   * Stops the current training process
   */
  const stopTraining = () => {
    sendMessage({ type: 'stop_training' });
    setIsTraining(false);
  };

  return {
    progress,
    isTraining,
    error,
    startTraining,
    stopTraining,
  };
};
