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
  reconstruction_loss: number;
  avg_pixel_error: number;
  loss_history: Array<{ epoch: number; train_loss: number; val_loss: number }>;
  original_images: number[][];
  reconstructed_images: number[][];
  latent_representations: Array<{ x: number; y: number; label: number }>;
  visualization_data: {
    n_samples: number;
    image_shape: number[];
    latent_dim: number;
    projection_method: string;
  };
  execution_time_ms: number;
  model_info: {
    latent_dim: number;
    hidden_dim: number;
    total_epochs: number;
    encoder_params: number;
    decoder_params: number;
  };
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
    const val = Math.round(pixels[i / 4] * 255);
    imageData.data[i] = val;
    imageData.data[i + 1] = val;
    imageData.data[i + 2] = val;
    imageData.data[i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL();
}

export function Visualization({ result }: VisualizationProps) {
  const { loss_history, original_images, reconstructed_images, latent_representations, visualization_data, model_info } = result;

  const size = visualization_data.image_shape[0];
  const numSamples = Math.min(5, original_images.length);

  const groupedLatentData = useMemo(() => {
    const groups: { [key: number]: any[] } = {};
    latent_representations.forEach((point) => {
      const label = point.label;
      if (!groups[label]) {
        groups[label] = [];
      }
      groups[label].push(point);
    });
    return groups;
  }, [latent_representations]);

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
      <div className="h-[400px] w-full">
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

      {/* Original vs Reconstructed Images */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">Original vs Reconstructed Images</h3>
        <div className="grid grid-cols-2 gap-4">
          {Array.from({ length: numSamples }).map((_, idx) => (
            <div key={idx} className="space-y-2">
              <div className="flex gap-4 justify-center items-start">
                <div className="text-center">
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Original</p>
                  <img
                    src={renderImage(original_images[idx], size)}
                    alt={`Original ${idx}`}
                    className="border-2 border-gray-300 dark:border-gray-600 rounded"
                    style={{ width: '100px', height: '100px', imageRendering: 'pixelated' }}
                  />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Reconstructed</p>
                  <img
                    src={renderImage(reconstructed_images[idx], size)}
                    alt={`Reconstructed ${idx}`}
                    className="border-2 border-gray-300 dark:border-gray-600 rounded"
                    style={{ width: '100px', height: '100px', imageRendering: 'pixelated' }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Latent Space Visualization */}
      {model_info.latent_dim === 2 && (
        <div className="h-[400px] w-full">
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
      )}

      {model_info.latent_dim > 2 && (
        <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
          <p className="text-sm text-yellow-900 dark:text-yellow-100">
            Latent space is {model_info.latent_dim}D. Showing t-SNE projection instead of raw space.
          </p>
        </div>
      )}

      {/* Model Info */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Model Configuration
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Latent Dim: {model_info.latent_dim}</p>
            <p>Hidden Dim: {model_info.hidden_dim}</p>
            <p>Encoder Params: {model_info.encoder_params.toLocaleString()}</p>
            <p>Decoder Params: {model_info.decoder_params.toLocaleString()}</p>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Quality Metrics
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Reconstruction Loss: {result.reconstruction_loss.toFixed(6)}</p>
            <p>Avg Pixel Error: {result.avg_pixel_error.toFixed(6)}</p>
            <p>Epochs: {model_info.total_epochs}</p>
            <p>Samples: {visualization_data.n_samples}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
