import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from 'recharts';

interface TrainingResult {
  success: boolean;
  variant: string;
  metrics: Record<string, number>;
  loss_history: Array<{ epoch: number; train_loss: number; val_loss: number }>;
  original_images: number[][];
  noisy_images?: number[][];
  reconstructed_images: number[][];
  latent_space: number[][];
  latent_labels: number[];
  reconstruction_errors: number[];
  learned_filters: number[][];
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
  parameters_used: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

const DIGIT_COLORS = [
  '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
  '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
];

function renderImage(pixels: number[], size: number) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  const imageData = ctx.createImageData(size, size);
  for (let i = 0; i < pixels.length && i < imageData.data.length; i += 4) {
    const val = Math.round(Math.min(1, Math.max(0, pixels[i / 4])) * 255);
    imageData.data[i] = val;
    imageData.data[i + 1] = val;
    imageData.data[i + 2] = val;
    imageData.data[i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL();
}

export function Visualization({ result }: VisualizationProps) {
  const { loss_history, original_images, noisy_images, reconstructed_images, latent_space, latent_labels, variant } = result;

  const size = 28;
  const numSamples = Math.min(3, original_images.length);
  const showNoisyImages = variant === 'denoising' && noisy_images && noisy_images.length > 0;

  const groupedLatentData = useMemo(() => {
    const groups: { [key: number]: any[] } = {};
    latent_space.forEach((point, idx) => {
      const label = latent_labels[idx];
      if (!groups[label]) {
        groups[label] = [];
      }
      groups[label].push({
        x: point[0],
        y: point[1],
        label: label,
      });
    });
    return groups;
  }, [latent_space, latent_labels]);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
          <p className="text-sm font-semibold mb-1">Digit: {data.label}</p>
          <p className="text-xs text-gray-600 dark:text-gray-400">X: {data.x.toFixed(3)}</p>
          <p className="text-xs text-gray-600 dark:text-gray-400">Y: {data.y.toFixed(3)}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Loss History */}
      <div className="h-[350px] w-full">
        <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">Training Loss</h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={loss_history}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
            <XAxis dataKey="epoch" className="text-gray-700 dark:text-gray-300" />
            <YAxis className="text-gray-700 dark:text-gray-300" />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="train_loss" stroke="#ef4444" strokeWidth={2} />
            <Line type="monotone" dataKey="val_loss" stroke="#3b82f6" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Images Comparison */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
          {showNoisyImages ? 'Original → Noisy → Reconstructed' : 'Original vs Reconstructed'}
        </h3>
        <div className="space-y-4">
          {Array.from({ length: numSamples }).map((_, idx) => (
            <div key={idx} className="flex gap-2 justify-center items-center">
              <div className="text-center">
                <p className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">Original</p>
                <img
                  src={renderImage(original_images[idx], size)}
                  alt={`Original ${idx}`}
                  className="border-2 border-gray-300 dark:border-gray-600 rounded"
                  style={{ width: '80px', height: '80px', imageRendering: 'pixelated' }}
                />
              </div>
              {showNoisyImages && (
                <div className="text-center">
                  <p className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">Noisy</p>
                  <img
                    src={renderImage(noisy_images![idx], size)}
                    alt={`Noisy ${idx}`}
                    className="border-2 border-orange-300 dark:border-orange-600 rounded"
                    style={{ width: '80px', height: '80px', imageRendering: 'pixelated' }}
                  />
                </div>
              )}
              <div className="text-center">
                <p className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">Reconstructed</p>
                <img
                  src={renderImage(reconstructed_images[idx], size)}
                  alt={`Reconstructed ${idx}`}
                  className="border-2 border-blue-300 dark:border-blue-600 rounded"
                  style={{ width: '80px', height: '80px', imageRendering: 'pixelated' }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Latent Space Visualization */}
      <div className="h-[350px] w-full">
        <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">Latent Space (2D)</h3>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
            <XAxis
              type="number"
              dataKey="x"
              name="Latent X"
              label={{ value: 'Latent X', position: 'bottom', offset: 40 }}
              className="text-gray-700 dark:text-gray-300"
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Latent Y"
              label={{ value: 'Latent Y', angle: -90, position: 'left', offset: 40 }}
              className="text-gray-700 dark:text-gray-300"
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
            {Object.entries(groupedLatentData).map(([label, data]) => (
              <Scatter
                key={label}
                name={`Digit ${label}`}
                data={data}
                fill={DIGIT_COLORS[parseInt(label)]}
                fillOpacity={0.6}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Training Metrics
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Final Loss: {result.metrics.final_loss?.toFixed(6) || 'N/A'}</p>
            <p>Reconstruction MSE: {result.metrics.reconstruction_mse?.toFixed(6) || 'N/A'}</p>
            <p>Avg Error: {result.metrics.avg_reconstruction_error?.toFixed(6) || 'N/A'}</p>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Configuration
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Variant: {result.variant}</p>
            <p>Latent Dim: {result.parameters_used.latent_dim}</p>
            <p>Epochs: {result.parameters_used.epochs}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
