import * as React from "react";
import {
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface BarChartProps {
  data: Array<Record<string, any>>;
  xKey: string;
  yKey: string | string[];
  title?: string;
  xLabel?: string;
  yLabel?: string;
  className?: string;
  height?: number;
  showGrid?: boolean;
  showLegend?: boolean;
  colors?: string[];
  children?: React.ReactNode;
}

const defaultColors = [
  "hsl(var(--primary))",
  "hsl(var(--secondary))",
  "hsl(var(--accent))",
  "#8884d8",
  "#82ca9d",
  "#ffc658",
  "#ff7c7c",
  "#a78bfa",
];

export function BarChart({
  data,
  xKey,
  yKey,
  title,
  xLabel,
  yLabel,
  className,
  height = 400,
  showGrid = true,
  showLegend = true,
  colors = defaultColors,
}: BarChartProps) {
  const yKeys = Array.isArray(yKey) ? yKey : [yKey];

  return (
    <Card className={className}>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent className={cn(title ? "" : "pt-6")}>
        <ResponsiveContainer width="100%" height={height}>
          <RechartsBarChart
            data={data}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            {showGrid && (
              <CartesianGrid
                strokeDasharray="3 3"
                className="stroke-muted"
                opacity={0.3}
              />
            )}
            <XAxis
              dataKey={xKey}
              label={xLabel ? { value: xLabel, position: "insideBottom", offset: -5 } : undefined}
              className="text-muted-foreground"
              tick={{ fill: "currentColor" }}
            />
            <YAxis
              label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft" } : undefined}
              className="text-muted-foreground"
              tick={{ fill: "currentColor" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--popover))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "var(--radius)",
              }}
              labelStyle={{ color: "hsl(var(--popover-foreground))" }}
            />
            {showLegend && <Legend />}
            {yKeys.map((key, index) => (
              <Bar
                key={key}
                dataKey={key}
                fill={colors[index % colors.length]}
              />
            ))}
          </RechartsBarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
