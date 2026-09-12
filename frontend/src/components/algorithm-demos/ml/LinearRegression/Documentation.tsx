import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface DocumentationProps {
  metadata?: {
    name: string;
    description: string;
    difficulty: string;
    tags: string[];
    use_cases: string[];
    complexity: {
      time: string;
      space: string;
    };
    theory: string;
    pros: string[];
    cons: string[];
    related_algorithms: string[];
  };
}

export function Documentation({ metadata }: DocumentationProps) {
  if (!metadata) {
    return null;
  }

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Algorithm Theory</CardTitle>
          <CardDescription>How Linear Regression works</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {metadata.theory}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Use Cases</CardTitle>
          <CardDescription>Common applications</CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {metadata.use_cases.map((useCase, idx) => (
              <li key={idx} className="text-sm text-muted-foreground flex items-start">
                <span className="mr-2">•</span>
                <span>{useCase}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Advantages</CardTitle>
          <CardDescription>Strengths of this algorithm</CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {metadata.pros.map((pro, idx) => (
              <li key={idx} className="text-sm text-muted-foreground flex items-start">
                <span className="mr-2 text-green-500">✓</span>
                <span>{pro}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Limitations</CardTitle>
          <CardDescription>Considerations and constraints</CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {metadata.cons.map((con, idx) => (
              <li key={idx} className="text-sm text-muted-foreground flex items-start">
                <span className="mr-2 text-yellow-500">⚠</span>
                <span>{con}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Complexity Analysis</CardTitle>
          <CardDescription>Performance characteristics</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <p className="text-sm font-medium mb-1">Time Complexity</p>
            <code className="text-sm bg-muted px-2 py-1 rounded">
              {metadata.complexity.time}
            </code>
            <p className="text-xs text-muted-foreground mt-1">
              Training time grows quadratically with the number of features
            </p>
          </div>
          <div>
            <p className="text-sm font-medium mb-1">Space Complexity</p>
            <code className="text-sm bg-muted px-2 py-1 rounded">
              {metadata.complexity.space}
            </code>
            <p className="text-xs text-muted-foreground mt-1">
              Memory usage grows linearly with the number of features
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Related Algorithms</CardTitle>
          <CardDescription>Similar or alternative approaches</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {metadata.related_algorithms.map((algo, idx) => (
              <Badge key={idx} variant="secondary">
                {algo.split('-').map(word =>
                  word.charAt(0).toUpperCase() + word.slice(1)
                ).join(' ')}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
