import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  metadata?: {
    theory: string;
    complexity: {
      time: string;
      space: string;
    };
    use_cases: string[];
    pros: string[];
    cons: string[];
    tags: string[];
  };
}

export function Documentation({ metadata }: DocumentationProps) {
  if (!metadata) {
    return null;
  }

  return (
    <div className="grid gap-6 md:grid-cols-2">
      {/* Theory */}
      <Card>
        <CardHeader>
          <CardTitle>Algorithm Theory</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground whitespace-pre-line">
            {metadata.theory}
          </p>
        </CardContent>
      </Card>

      {/* Complexity */}
      <Card>
        <CardHeader>
          <CardTitle>Complexity Analysis</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="text-sm font-medium mb-1">Time Complexity</p>
            <code className="text-sm bg-muted px-2 py-1 rounded">
              {metadata.complexity.time}
            </code>
            <p className="text-xs text-muted-foreground mt-2">
              Where n is the number of samples, d is the number of features, k is the number
              of trees, and depth is the maximum tree depth.
            </p>
          </div>
          <div>
            <p className="text-sm font-medium mb-1">Space Complexity</p>
            <code className="text-sm bg-muted px-2 py-1 rounded">
              {metadata.complexity.space}
            </code>
            <p className="text-xs text-muted-foreground mt-2">
              Where k is the number of trees and n is the number of samples.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Use Cases */}
      <Card>
        <CardHeader>
          <CardTitle>Use Cases</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {metadata.use_cases.map((useCase, index) => (
              <li key={index} className="flex items-start">
                <span className="mr-2 mt-1 h-1.5 w-1.5 rounded-full bg-primary flex-shrink-0" />
                <span className="text-sm text-muted-foreground">{useCase}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* Pros and Cons */}
      <Card>
        <CardHeader>
          <CardTitle>Advantages & Limitations</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="text-sm font-medium mb-2 text-green-600 dark:text-green-400">
              Advantages
            </p>
            <ul className="space-y-1">
              {metadata.pros.map((pro, index) => (
                <li key={index} className="flex items-start">
                  <span className="mr-2 text-green-600 dark:text-green-400">+</span>
                  <span className="text-sm text-muted-foreground">{pro}</span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <p className="text-sm font-medium mb-2 text-orange-600 dark:text-orange-400">
              Limitations
            </p>
            <ul className="space-y-1">
              {metadata.cons.map((con, index) => (
                <li key={index} className="flex items-start">
                  <span className="mr-2 text-orange-600 dark:text-orange-400">-</span>
                  <span className="text-sm text-muted-foreground">{con}</span>
                </li>
              ))}
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
