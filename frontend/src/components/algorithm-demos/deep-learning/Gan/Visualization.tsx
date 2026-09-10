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
} from 'recharts';

interface TrainingResult {
  loss_history: Array<{
    epoch: number;
    g_loss: number;
    d_loss: number;
    d_real_loss?: number;
    d_fake_loss?: number;
    d_real_accuracy?: number;
    d_fake_accuracy?: number;
  }>;
  generated_samples: Array<{
    epoch: number;
    samples: number[][];
  }>;
  final_samples: number[][];
  interpolated_samples?: number[][];
  decision_boundary?: Record<string, any>;
  visualization_data: Record<string, any>;
  execution_time_ms: number;
  model_info: Record<string, any>;
}

interface VisualizationProps {
  result: TrainingResult;
}

function renderImage(pixels: number[], size: number = 8) {
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
  const { loss_history, generated_samples, final_samples } = result;

  const chartData = useMemo(() => {
    return loss_history.map((item) => ({
      epoch: item.epoch,
      g_loss: item.g_loss,
      d_loss: item.d_loss,
      d_real_loss: item.d_real_loss || item.d_loss / 2,
      d_fake_loss: item.d_fake_loss || item.d_loss / 2,
    }));
  }, [loss_history]);

  // Select samples from key epochs
  const samplesPerEpoch = useMemo(() => {
    const selected = [];
    for (let i = 0; i < generated_samples.length; i += Math.ceil(generated_samples.length / 4)) {
      if (generated_samples[i]) {
        selected.push(generated_samples[i]);
      }
    }
    // Also add the last one
    if (selected.length === 0 || selected[selected.length - 1].epoch !== generated_samples[generated_samples.length - 1].epoch) {
      selected.push(generated_samples[generated_samples.length - 1]);
    }
    return selected.slice(0, 4);
  }, [generated_samples]);

  const numFinalSamples = Math.min(16, final_samples.length);

  return (
    <div className="space-y-6">
      {/* Loss History */}
      <div className="h-[350px] w-full">
        <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">Training Loss</h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
            <XAxis dataKey="epoch" className="text-gray-700 dark:text-gray-300" />
            <YAxis className="text-gray-700 dark:text-gray-300" />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="g_loss" stroke="#10b981" strokeWidth={2} name="Generator Loss" />
            <Line type="monotone" dataKey="d_loss" stroke="#ef4444" strokeWidth={2} name="Discriminator Loss" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Generated Samples at Different Epochs */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">Generation Progress</h3>
        <div className="grid grid-cols-4 gap-4">
          {samplesPerEpoch.map((epochData) => (
            <div key={epochData.epoch} className="text-center">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                Epoch {epochData.epoch}
              </p>
              <div className="grid grid-cols-2 gap-2">
                {epochData.samples.slice(0, 4).map((sample, idx) => (
                  <img
                    key={idx}
                    src={renderImage(sample, 8)}
                    alt={`Generated ${epochData.epoch}-${idx}`}
                    className="border border-gray-300 dark:border-gray-600 rounded"
                    style={{ width: '60px', height: '60px', imageRendering: 'pixelated' }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Final Generated Samples */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">Final Generated Samples</h3>
        <div className="grid grid-cols-8 gap-2">
          {Array.from({ length: numFinalSamples }).map((_, idx) => (
            <img
              key={idx}
              src={renderImage(final_samples[idx], 8)}
              alt={`Final ${idx}`}
              className="border-2 border-green-400 dark:border-green-600 rounded hover:shadow-lg transition-shadow"
              style={{ width: '70px', height: '70px', imageRendering: 'pixelated' }}
            />
          ))}
        </div>
      </div>

      {/* Model Statistics */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Generator Architecture
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Input: {result.model_info.latent_dim || 100}D noise vector</p>
            <p>Hidden: {result.model_info.g_hidden || 'N/A'}</p>
            <p>Output: 64 pixels (8×8)</p>
            <p>Parameters: {result.model_info.generator_params?.toLocaleString() || 'N/A'}</p>
          </div>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Discriminator Architecture
          </h4>
          <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
            <p>Input: 64 pixels (8×8)</p>
            <p>Hidden: {result.model_info.d_hidden || 'N/A'}</p>
            <p>Output: Binary classification</p>
            <p>Parameters: {result.model_info.discriminator_params?.toLocaleString() || 'N/A'}</p>
          </div>
        </div>
      </div>

      {/* Training Info */}
      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-sm text-blue-900 dark:text-blue-100">
        <p className="font-semibold mb-2">Training Notes</p>
        <ul className="text-xs space-y-1 list-disc list-inside">
          <li>Generator learns to fool the discriminator by creating realistic samples</li>
          <li>Discriminator learns to distinguish real from fake samples</li>
          <li>Training success depends heavily on learning rate and architecture balance</li>
          <li>Mode collapse can occur if generator converges to a few similar samples</li>
        </ul>
      </div>
    </div>
  );
}
