import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface DocumentationProps {
  metadata?: {
    theory?: string;
    complexity?: {
      time: string;
      space: string;
    };
    pros?: string[];
    cons?: string[];
    use_cases?: string[];
    tags?: string[];
    difficulty?: string;
  };
}

export function Documentation({ metadata }: DocumentationProps) {
  if (!metadata) {
    return null;
  }

  return (
    <div className="space-y-6">
      {metadata.difficulty && (
        <div>
          <h3 className="text-lg font-semibold mb-2">Difficulty Level</h3>
          <Badge variant="outline" className="text-sm">
            {metadata.difficulty}
          </Badge>
        </div>
      )}

      {metadata.tags && metadata.tags.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-2">Tags</h3>
          <div className="flex flex-wrap gap-2">
            {metadata.tags.map((tag) => (
              <Badge key={tag} variant="secondary">
                {tag}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {metadata.theory && (
        <Card>
          <CardHeader>
            <CardTitle>How It Works</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {metadata.theory}
            </p>
          </CardContent>
        </Card>
      )}

      {metadata.complexity && (
        <Card>
          <CardHeader>
            <CardTitle>Complexity Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div>
                <span className="font-medium text-sm">Time Complexity:</span>{' '}
                <code className="text-sm bg-muted px-2 py-1 rounded">
                  {metadata.complexity.time}
                </code>
              </div>
              <div>
                <span className="font-medium text-sm">Space Complexity:</span>{' '}
                <code className="text-sm bg-muted px-2 py-1 rounded">
                  {metadata.complexity.space}
                </code>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                where n is the number of training samples, and the complexity depends on the kernel used.
                Training complexity ranges from O(n²) for linear kernels to O(n³) for non-linear kernels.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {metadata.use_cases && metadata.use_cases.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Common Use Cases</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {metadata.use_cases.map((useCase, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-primary mt-1">•</span>
                  <span className="text-sm text-muted-foreground">{useCase}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      <div className="grid md:grid-cols-2 gap-4">
        {metadata.pros && metadata.pros.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-green-600 dark:text-green-400">
                Advantages
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {metadata.pros.map((pro, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-green-600 dark:text-green-400 mt-1">✓</span>
                    <span className="text-sm text-muted-foreground">{pro}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        {metadata.cons && metadata.cons.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-orange-600 dark:text-orange-400">
                Limitations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {metadata.cons.map((con, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-orange-600 dark:text-orange-400 mt-1">⚠</span>
                    <span className="text-sm text-muted-foreground">{con}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Key Concepts</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h4 className="font-medium text-sm mb-1">Support Vectors</h4>
              <p className="text-sm text-muted-foreground">
                Data points that lie closest to the decision boundary. These points are critical
                as they define the position and orientation of the hyperplane. The model only
                uses these points, making it memory efficient.
              </p>
            </div>
            <div>
              <h4 className="font-medium text-sm mb-1">Kernel Trick</h4>
              <p className="text-sm text-muted-foreground">
                Transforms data into higher dimensions where linear separation becomes possible.
                Different kernels (linear, polynomial, RBF, sigmoid) suit different data patterns.
              </p>
            </div>
            <div>
              <h4 className="font-medium text-sm mb-1">Margin Maximization</h4>
              <p className="text-sm text-muted-foreground">
                SVM finds the hyperplane that maximizes the margin (distance) between classes.
                A larger margin generally leads to better generalization on unseen data.
              </p>
            </div>
            <div>
              <h4 className="font-medium text-sm mb-1">Regularization (C)</h4>
              <p className="text-sm text-muted-foreground">
                The C parameter controls the trade-off between achieving a low training error
                and a low testing error. Smaller C creates a wider margin with more
                misclassifications, while larger C aims for fewer misclassifications.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
