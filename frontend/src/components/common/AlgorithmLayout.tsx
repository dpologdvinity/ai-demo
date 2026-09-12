import * as React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface AlgorithmLayoutProps {
  title: string;
  description?: string;
  children: React.ReactNode;
  sections?: {
    parameters?: React.ReactNode;
    visualization?: React.ReactNode;
    code?: React.ReactNode;
    results?: React.ReactNode;
  };
  className?: string;
}

export function AlgorithmLayout({
  title,
  description,
  children,
  sections,
  className,
}: AlgorithmLayoutProps) {
  if (!sections) {
    // Simple layout - just render children
    return (
      <div className={cn("container mx-auto p-6 space-y-6", className)}>
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">{title}</h1>
          {description && (
            <p className="text-muted-foreground">{description}</p>
          )}
        </div>
        {children}
      </div>
    );
  }

  // Structured layout with sections
  return (
    <div className={cn("container mx-auto p-6 space-y-6", className)}>
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">{title}</h1>
        {description && <p className="text-muted-foreground">{description}</p>}
      </div>

      {/* Main content area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column - Parameters and Code */}
        <div className="space-y-6">
          {sections.parameters && (
            <Card>
              <CardHeader>
                <CardTitle>Parameters</CardTitle>
                <CardDescription>
                  Adjust the algorithm parameters
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {sections.parameters}
              </CardContent>
            </Card>
          )}

          {sections.code && (
            <Card>
              <CardHeader>
                <CardTitle>Implementation</CardTitle>
                <CardDescription>Algorithm source code</CardDescription>
              </CardHeader>
              <CardContent className="p-0">{sections.code}</CardContent>
            </Card>
          )}
        </div>

        {/* Right column - Visualization and Results */}
        <div className="lg:col-span-2 space-y-6">
          {sections.visualization && (
            <Card>
              <CardHeader>
                <CardTitle>Visualization</CardTitle>
                <CardDescription>
                  Interactive algorithm visualization
                </CardDescription>
              </CardHeader>
              <CardContent>{sections.visualization}</CardContent>
            </Card>
          )}

          {sections.results && (
            <Card>
              <CardHeader>
                <CardTitle>Results</CardTitle>
                <CardDescription>Algorithm output and metrics</CardDescription>
              </CardHeader>
              <CardContent>{sections.results}</CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Additional content */}
      {children}
    </div>
  );
}
