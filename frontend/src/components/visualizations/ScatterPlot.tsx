import * as React from "react";
import {
  ScatterChart as RechartsScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ZAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface ScatterPlotProps {
  data: Array<Record<string, any>> | Array<Array<Record<string, any>>>;
  xKey: string;
  yKey: string;
  colorKey?: string;
  title?: string;
  xLabel?: string;
  yLabel?: string;
  className?: string;
  height?: number;
  showGrid?: boolean;
  showLegend?: boolean;
  colors?: string[];
  clusterNames?: string[];
}

const defaultColors = [
  "#8884d8",
  "#82ca9d",
  "#ffc658",
  "#ff7c7c",
  "#a78bfa",
  "#fb923c",
  "#38bdf8",
  "#4ade80",
];

export function ScatterPlot({
  data,
  xKey,
  yKey,
  colorKey,
  title,
  xLabel,
  yLabel,
  className,
  height = 400,
  showGrid = true,
  showLegend = true,
  colors = defaultColors,
  clusterNames,
}: ScatterPlotProps) {
  // Handle both single dataset and clustered data
  const isClusteredData = Array.isArray(data[0]) || colorKey;
  const datasets = isClusteredData
    ? Array.isArray(data[0])
      ? (data as Array<Array<Record<string, any>>>)
      : groupByColor(data as Array<Record<string, any>>, colorKey!)
    : [data as Array<Record<string, any>>];

  return (
    <Card className={className}>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent className={cn(title ? "" : "pt-6")}>
        <ResponsiveContainer width="100%" height={height}>
          <RechartsScatterChart
            margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
          >
            {showGrid && (
              <CartesianGrid
                strokeDasharray="3 3"
                className="stroke-muted"
                opacity={0.3}
              />
            )}
            <XAxis
              type="number"
              dataKey={xKey}
              name={xLabel || xKey}
              label={xLabel ? { value: xLabel, position: "insideBottom", offset: -10 } : undefined}
              className="text-muted-foreground"
              tick={{ fill: "currentColor" }}
            />
            <YAxis
              type="number"
              dataKey={yKey}
              name={yLabel || yKey}
              label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft" } : undefined}
              className="text-muted-foreground"
              tick={{ fill: "currentColor" }}
            />
            <ZAxis range={[60, 60]} />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              contentStyle={{
                backgroundColor: "hsl(var(--popover))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "var(--radius)",
              }}
              labelStyle={{ color: "hsl(var(--popover-foreground))" }}
            />
            {showLegend && <Legend />}
            {datasets.map((dataset, index) => (
              <Scatter
                key={index}
                name={
                  clusterNames?.[index] ||
                  (isClusteredData ? `Cluster ${index + 1}` : "Data")
                }
                data={dataset}
                fill={colors[index % colors.length]}
              />
            ))}
          </RechartsScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function groupByColor(
  data: Array<Record<string, any>>,
  colorKey: string
): Array<Array<Record<string, any>>> {
  const groups = new Map<any, Array<Record<string, any>>>();

  data.forEach((item) => {
    const color = item[colorKey];
    if (!groups.has(color)) {
      groups.set(color, []);
    }
    groups.get(color)!.push(item);
  });

  return Array.from(groups.values());
}
