import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/common/Card';
import { Book, CheckCircle, XCircle, Lightbulb } from 'lucide-react';

interface DocumentationProps {
  metadata?: {
    theory?: string;
    pros?: string[];
    cons?: string[];
    use_cases?: string[];
    related_algorithms?: string[];
  };
}

export function Documentation({ metadata }: DocumentationProps) {
  if (!metadata) return null;

  return (
    <div className="space-y-6">
      {/* Theory Section */}
      {metadata.theory && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Book className="w-5 h-5 text-primary" />
              <CardTitle>Algorithm Theory</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="prose prose-sm max-w-none dark:prose-invert">
              {metadata.theory.split('\n\n').map((paragraph, idx) => (
                <p key={idx} className="text-sm text-muted-foreground mb-3">
                  {paragraph}
                </p>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Use Cases */}
      {metadata.use_cases && metadata.use_cases.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Lightbulb className="w-5 h-5 text-yellow-500" />
              <CardTitle>Use Cases</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {metadata.use_cases.map((useCase, idx) => (
                <li
                  key={idx}
                  className="flex items-start gap-2 text-sm text-muted-foreground"
                >
                  <span className="text-primary mt-0.5">•</span>
                  {useCase}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Pros and Cons */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Pros */}
        {metadata.pros && metadata.pros.length > 0 && (
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <CardTitle>Advantages</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {metadata.pros.map((pro, idx) => (
                  <li
                    key={idx}
                    className="flex items-start gap-2 text-sm text-muted-foreground"
                  >
                    <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                    {pro}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        {/* Cons */}
        {metadata.cons && metadata.cons.length > 0 && (
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <XCircle className="w-5 h-5 text-red-500" />
                <CardTitle>Limitations</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {metadata.cons.map((con, idx) => (
                  <li
                    key={idx}
                    className="flex items-start gap-2 text-sm text-muted-foreground"
                  >
                    <XCircle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                    {con}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Related Algorithms */}
      {metadata.related_algorithms && metadata.related_algorithms.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Related Algorithms</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {metadata.related_algorithms.map((algo, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1 bg-muted rounded-full text-sm capitalize"
                >
                  {algo.replace(/-/g, ' ')}
                </span>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
