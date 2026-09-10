import * as React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export interface ConfusionMatrixProps {
  matrix: number[][];
  labels: string[];
  title?: string;
  className?: string;
}

export function ConfusionMatrix({
  matrix,
  labels,
  title = "Confusion Matrix",
  className,
}: ConfusionMatrixProps) {
  const maxValue = Math.max(...matrix.flat());
  const cellSize = 80;

  const getColor = (value: number): string => {
    const intensity = value / maxValue;
    return `hsl(var(--primary) / ${intensity * 0.8 + 0.2})`;
  };

  const getTextColor = (value: number): string => {
    const intensity = value / maxValue;
    return intensity > 0.5 ? "white" : "hsl(var(--foreground))";
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            {/* Column labels */}
            <div className="flex mb-2">
              <div style={{ width: cellSize }} className="flex-shrink-0" />
              <div className="flex gap-1">
                {labels.map((label, index) => (
                  <div
                    key={index}
                    style={{ width: cellSize }}
                    className="text-center text-sm font-medium text-muted-foreground flex-shrink-0"
                  >
                    {label}
                  </div>
                ))}
              </div>
            </div>

            {/* Matrix rows */}
            {matrix.map((row, rowIndex) => (
              <div key={rowIndex} className="flex gap-1 mb-1">
                {/* Row label */}
                <div
                  style={{ width: cellSize }}
                  className="flex items-center justify-end pr-2 text-sm font-medium text-muted-foreground flex-shrink-0"
                >
                  {labels[rowIndex]}
                </div>

                {/* Matrix cells */}
                {row.map((value, colIndex) => (
                  <div
                    key={colIndex}
                    style={{
                      width: cellSize,
                      height: cellSize,
                      backgroundColor: getColor(value),
                      color: getTextColor(value),
                    }}
                    className="flex items-center justify-center rounded border border-border text-sm font-semibold flex-shrink-0"
                  >
                    {value}
                  </div>
                ))}
              </div>
            ))}

            {/* Legend */}
            <div className="mt-4 flex items-center gap-4 text-xs text-muted-foreground">
              <span>Predicted →</span>
              <span>← Actual</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
