import { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export interface MathPanelProps {
  title?: string; // e.g. "Update Rule"
  formula: string; // LaTeX source, e.g. "\\theta = \\theta - \\alpha \\nabla L(\\theta)"
  substitution?: string; // LaTeX with actual numbers plugged in
  note?: string; // optional short plain-text explanation, one line
  className?: string;
}

/**
 * MathPanel: Renders LaTeX formulas using KaTeX with optional substitution.
 */
export function MathPanel({
  title = 'Formula',
  formula,
  substitution,
  note,
  className,
}: MathPanelProps) {
  // Memoize KaTeX renders to avoid recomputing on every render
  const formulaHtml = useMemo(
    () =>
      katex.renderToString(formula, {
        throwOnError: false,
        displayMode: true,
      }),
    [formula]
  );

  const substitutionHtml = useMemo(
    () =>
      substitution
        ? katex.renderToString(substitution, {
            throwOnError: false,
            displayMode: true,
          })
        : null,
    [substitution]
  );

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Main formula */}
        <div
          className="overflow-x-auto"
          dangerouslySetInnerHTML={{ __html: formulaHtml }}
        />

        {/* Substitution with numbers (live values) */}
        {substitutionHtml && (
          <div
            className="overflow-x-auto text-neon-cyan"
            dangerouslySetInnerHTML={{ __html: substitutionHtml }}
          />
        )}

        {/* Optional note */}
        {note && <p className="text-sm text-muted-foreground">{note}</p>}
      </CardContent>
    </Card>
  );
}
