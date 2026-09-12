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
          <CardDescription>How K-Nearest Neighbors works</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground leading-relaxed space-y-3">
            {metadata.theory.split('\n\n').map((paragraph, idx) => (
              <p key={idx}>{paragraph}</p>
            ))}
          </div>
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
              Where n is the number of samples and d is the number of features.
              Prediction requires computing distance to all training samples.
            </p>
          </div>
          <div>
            <p className="text-sm font-medium mb-1">Space Complexity</p>
            <code className="text-sm bg-muted px-2 py-1 rounded">
              {metadata.complexity.space}
            </code>
            <p className="text-xs text-muted-foreground mt-1">
              KNN stores all training data, so memory usage grows with dataset size
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

      <Card className="md:col-span-2">
        <CardHeader>
          <CardTitle>Key Concepts</CardTitle>
          <CardDescription>Important ideas to understand</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4 text-sm">
            <div>
              <h4 className="font-medium mb-1">Choosing K</h4>
              <p className="text-muted-foreground">
                The value of K significantly impacts model performance. Small K values
                (e.g., K=1) can lead to overfitting and sensitivity to noise, while
                large K values may oversimplify the decision boundary. Odd values of K
                are typically preferred for binary classification to avoid ties.
              </p>
            </div>
            <div>
              <h4 className="font-medium mb-1">Distance Metrics</h4>
              <p className="text-muted-foreground">
                Different metrics capture different notions of similarity. Euclidean
                distance works well for continuous features, Manhattan distance is
                more robust to outliers, and Minkowski generalizes both approaches.
              </p>
            </div>
            <div>
              <h4 className="font-medium mb-1">Feature Scaling</h4>
              <p className="text-muted-foreground">
                KNN is highly sensitive to feature scales because it relies on distance
                calculations. Features with larger ranges will dominate the distance
                computation, so normalization or standardization is essential.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
