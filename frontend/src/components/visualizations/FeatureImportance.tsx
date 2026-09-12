import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface FeatureImportanceProps {
  data: Record<string, number>;
  title?: string;
  className?: string;
}

export const FeatureImportance: React.FC<FeatureImportanceProps> = ({
  data,
  title = 'Feature Importance',
  className = ''
}) => {
  // Convert data object to array and sort by importance
  const chartData = Object.entries(data)
    .map(([name, value]) => ({
      name: name.replace(/_/g, ' '),
      importance: value,
      originalName: name
    }))
    .sort((a, b) => b.importance - a.importance);

  // Color gradient based on importance
  const getColor = (index: number, total: number) => {
    const hue = 210; // Blue hue
    const saturation = 70;
    const lightness = 45 + (index / total) * 20;
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  };

  return (
    <div className={`w-full ${className}`}>
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            domain={[0, 'dataMax']}
            tickFormatter={(value) => value.toFixed(3)}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={110}
            tick={{ fontSize: 12 }}
          />
          <Tooltip
            formatter={(value: number) => value.toFixed(4)}
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
          />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getColor(index, chartData.length)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-4 text-sm text-gray-600">
        <p>
          <strong>Top 3 Features:</strong>
        </p>
        <ul className="list-disc list-inside mt-2">
          {chartData.slice(0, 3).map((feature, idx) => (
            <li key={idx}>
              {feature.name}: {(feature.importance * 100).toFixed(2)}%
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default FeatureImportance;
