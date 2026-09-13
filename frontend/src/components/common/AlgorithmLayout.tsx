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
  children?: React.ReactNode;
  sections?: {
    parameters?: React.ReactNode;
    visualization?: React.ReactNode;
    code?: React.ReactNode;
    results?: React.ReactNode;
    theory?: React.ReactNode;
  };
  difficulty?: string;
  complexity?: string;
  tags?: string[];
  algorithmInfo?: any;
  category?: string;
  className?: string;
}

type Accent = "cyan" | "violet" | "green" | "amber" | "magenta";

const ACCENT_STYLES: Record<Accent, { text: string; border: string; bg: string }> = {
  cyan: { text: "text-neon-cyan", border: "border-neon-cyan/40", bg: "bg-neon-cyan/10" },
  violet: { text: "text-neon-violet", border: "border-neon-violet/40", bg: "bg-neon-violet/10" },
  green: { text: "text-neon-green", border: "border-neon-green/40", bg: "bg-neon-green/10" },
  amber: { text: "text-neon-amber", border: "border-neon-amber/40", bg: "bg-neon-amber/10" },
  magenta: { text: "text-neon-magenta", border: "border-neon-magenta/40", bg: "bg-neon-magenta/10" },
};

function accentForCategory(category?: string): Accent {
  const c = (category || "").toLowerCase();
  if (c.includes("deep learning")) return "violet";
  if (c.includes("natural language") || c === "nlp") return "green";
  if (c.includes("computer vision")) return "amber";
  if (c.includes("reinforcement")) return "magenta";
  return "cyan"; // machine learning and unspecified default
}

function AlgorithmHeader({
  title,
  description,
  category,
  difficulty,
}: Pick<AlgorithmLayoutProps, "title" | "description" | "category" | "difficulty">) {
  const accent = ACCENT_STYLES[accentForCategory(category)];
  return (
    <div className="space-y-3">
      {(category || difficulty) && (
        <div className="flex flex-wrap items-center gap-2 text-xs font-medium">
          {category && (
            <span className={cn("rounded-sm border px-2 py-0.5", accent.border, accent.bg, accent.text)}>
              {category}
            </span>
          )}
          {difficulty && (
            <span className="rounded-sm border border-border px-2 py-0.5 text-muted-foreground">
              {difficulty}
            </span>
          )}
        </div>
      )}
      <h1 className={cn("font-display text-3xl font-bold tracking-wide", accent.text)}>
        {title}
      </h1>
      {description && (
        <p className="max-w-[70ch] text-muted-foreground">{description}</p>
      )}
    </div>
  );
}

export function AlgorithmLayout({
  title,
  description,
  children,
  sections,
  category,
  difficulty,
  className,
}: AlgorithmLayoutProps) {
  if (!sections) {
    // Simple layout - just render children
    return (
      <div className={cn("container mx-auto p-6 space-y-6", className)}>
        <AlgorithmHeader
          title={title}
          description={description}
          category={category}
          difficulty={difficulty}
        />
        {children}
      </div>
    );
  }

  // Structured layout with sections
  return (
    <div className={cn("container mx-auto p-6 space-y-6", className)}>
      {/* Header */}
      <AlgorithmHeader
        title={title}
        description={description}
        category={category}
        difficulty={difficulty}
      />

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
