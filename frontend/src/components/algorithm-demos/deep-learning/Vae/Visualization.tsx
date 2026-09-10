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
  metrics?: Record<string, number>;
  original_images?: number[][];
  reconstructed_images?: number[][];
  generated_images?: number[][];
  latent_space?: number[][];
  latent_labels?: number[];
  loss_history?: Record<string, number[]>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info?: Record<string, any>;
  parameters_used: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

const DIGIT_COLORS = [
  '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
  '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
];

function renderImage(pixels: number[], size: number = 28) {
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
  const { loss_history, original_images, reconstructed_images, generated_images, latent_space, latent_labels } = result;

  const numSamples = Math.min(4, (original_images || []).length);
  const numGenerated = Math.min(8, (generated_images || []).length);

  const lossData = useMemo(() => {
    if (!loss_history) return [];
    const numEpochs = Object.values(loss_history)[0]?.length || 0;
    return Array.from({ length: numEpochs }).map((_, i) => ({
      epoch: i + 1,
      reconstruction_loss: loss_history['reconstruction_loss']?.[i] || 0,
      kl_divergence: loss_history['kl_divergence']?.[i] || 0,
    }));
  }, [loss_history]);

  const groupedLatentData = useMemo(() => {
    if (!latent_space || !latent_labels) return {};
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
      {lossData.length > 0 && (
        <div className="h-[350px] w-full">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">Training Loss</h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
              <XAxis dataKey="epoch" className="text-gray-700 dark:text-gray-300" />
              <YAxis className="text-gray-700 dark:text-gray-300" />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="reconstruction_loss" stroke="#ef4444" strokeWidth={2} />
              <Line type="monotone" dataKey="kl_divergence" stroke="#3b82f6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Original vs Reconstructed Images */}
      {original_images && original_images.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">Original vs Reconstructed</h3>
          <div className="flex gap-4 justify-center flex-wrap">
            {Array.from({ length: numSamples }).map((_, idx) => (
              <div key={idx} className="text-center">
                <div className="flex gap-2 mb-2">
                  <img
                    src={renderImage(original_images[idx])}
                    alt={`Original ${idx}`}
                    className="border-2 border-gray-300 dark:border-gray-600 rounded"
                    style={{ width: '70px', height: '70px', imageRendering: 'pixelated' }}
                  />
                  <img
                    src={renderImage(reconstructed_images![idx])}
                    alt={`Reconstructed ${idx}`}
                    className="border-2 border-blue-300 dark:border-blue-600 rounded"
                    style={{ width: '70px', height: '70px', imageRendering: 'pixelated' }}
                  />
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Orig → Recon</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Generated Samples */}
      {generated_images && generated_images.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">Generated Samples (Random Latent)</h3>
          <div className="flex gap-2 justify-center flex-wrap">
            {Array.from({ length: numGenerated }).map((_, idx) => (
              <img
                key={idx}
                src={renderImage(generated_images[idx])}
                alt={`Generated ${idx}`}
                className="border-2 border-green-300 dark:border-green-600 rounded"
                style={{ width: '70px', height: '70px', imageRendering: 'pixelated' }}
              />
            ))}
          </div>
        </div>
      )}

      {/* Latent Space Visualization */}
      {latent_space && latent_space.length > 0 && Object.keys(groupedLatentData).length > 0 && (
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
      )}

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Loss Components
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Final Loss: {result.metrics?.final_loss?.toFixed(2) || 'N/A'}</p>
            <p>Reconstruction: {result.metrics?.final_reconstruction_loss?.toFixed(2) || 'N/A'}</p>
            <p>KL Divergence: {result.metrics?.final_kl_divergence?.toFixed(4) || 'N/A'}</p>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Configuration
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Latent Dim: {result.parameters_used.latent_dim}</p>
            <p>Beta: {result.parameters_used.beta}</p>
            <p>Epochs: {result.parameters_used.epochs}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
