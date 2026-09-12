import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
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
  if (!metadata) return null;

  return (
    <div className="space-y-6">
      {/* Overview */}
      <Card>
        <CardHeader>
          <CardTitle>Algorithm Overview</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h3 className="font-semibold mb-2">Description</h3>
            <p className="text-sm text-muted-foreground">{metadata.description}</p>
          </div>

          <div>
            <h3 className="font-semibold mb-2">Difficulty</h3>
            <Badge variant="outline">{metadata.difficulty}</Badge>
          </div>

          <div>
            <h3 className="font-semibold mb-2">Tags</h3>
            <div className="flex flex-wrap gap-2">
              {metadata.tags.map((tag, idx) => (
                <Badge key={idx} variant="secondary">
                  {tag}
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Theory */}
      <Card>
        <CardHeader>
          <CardTitle>Theory</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {metadata.theory}
          </p>
        </CardContent>
      </Card>

      {/* Use Cases */}
      <Card>
        <CardHeader>
          <CardTitle>Use Cases</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
            {metadata.use_cases.map((useCase, idx) => (
              <li key={idx}>{useCase}</li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* Complexity */}
      <Card>
        <CardHeader>
          <CardTitle>Complexity Analysis</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div>
            <span className="font-semibold text-sm">Time Complexity: </span>
            <code className="text-sm bg-muted px-2 py-1 rounded">
              {metadata.complexity.time}
            </code>
          </div>
          <div>
            <span className="font-semibold text-sm">Space Complexity: </span>
            <code className="text-sm bg-muted px-2 py-1 rounded">
              {metadata.complexity.space}
            </code>
          </div>
        </CardContent>
      </Card>

      {/* Pros and Cons */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-green-600">Advantages</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
              {metadata.pros.map((pro, idx) => (
                <li key={idx}>{pro}</li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-red-600">Limitations</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
              {metadata.cons.map((con, idx) => (
                <li key={idx}>{con}</li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>

      {/* Related Algorithms */}
      <Card>
        <CardHeader>
          <CardTitle>Related Algorithms</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {metadata.related_algorithms.map((algo, idx) => (
              <Badge key={idx} variant="outline">
                {algo}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
