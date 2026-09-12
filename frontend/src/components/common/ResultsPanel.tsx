import * as React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface Metric {
  label: string;
  value: string | number;
  description?: string;
  format?: (value: number) => string;
}

export interface ResultsPanelProps {
  metrics?: Metric[];
  predictions?: Array<{
    label: string;
    value: string | number;
    confidence?: number;
  }>;
  visualizations?: React.ReactNode;
  className?: string;
}

export function ResultsPanel({
  metrics = [],
  predictions = [],
  visualizations,
  className,
}: ResultsPanelProps) {
  const formatValue = (metric: Metric): string => {
    if (typeof metric.value === "string") {
      return metric.value;
    }
    if (metric.format) {
      return metric.format(metric.value);
    }
    if (Number.isInteger(metric.value)) {
      return metric.value.toString();
    }
    return metric.value.toFixed(4);
  };

  return (
    <div className={cn("space-y-6", className)}>
      {/* Metrics Grid */}
      {metrics.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {metrics.map((metric, index) => (
            <Card key={index}>
              <CardHeader className="pb-2">
                <CardDescription>{metric.label}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{formatValue(metric)}</div>
                {metric.description && (
                  <p className="text-xs text-muted-foreground mt-1">
                    {metric.description}
                  </p>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Predictions */}
      {predictions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Predictions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {predictions.map((prediction, index) => (
                <div key={index} className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">
                      {prediction.label}
                    </span>
                    <span className="text-sm text-muted-foreground">
                      {prediction.value}
                    </span>
                  </div>
                  {prediction.confidence !== undefined && (
                    <div className="w-full bg-secondary rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full transition-all"
                        style={{
                          width: `${prediction.confidence * 100}%`,
                        }}
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Additional Visualizations */}
      {visualizations && (
        <Card>
          <CardContent className="pt-6">{visualizations}</CardContent>
        </Card>
      )}
    </div>
  );
}
