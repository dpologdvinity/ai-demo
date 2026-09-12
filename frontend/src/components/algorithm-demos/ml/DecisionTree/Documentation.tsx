import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

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
    related_algorithms?: string[];
  };
}

export function Documentation({ metadata }: DocumentationProps) {
  if (!metadata) {
    return (
      <div className="text-center text-muted-foreground py-8">
        Loading documentation...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Algorithm Theory</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm leading-relaxed text-muted-foreground">
            {metadata.theory}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Complexity Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex items-start gap-2">
              <span className="font-semibold text-sm min-w-24">Time:</span>
              <code className="text-sm bg-secondary px-2 py-1 rounded">
                {metadata.complexity?.time}
              </code>
            </div>
            <div className="flex items-start gap-2">
              <span className="font-semibold text-sm min-w-24">Space:</span>
              <code className="text-sm bg-secondary px-2 py-1 rounded">
                {metadata.complexity?.space}
              </code>
            </div>
            <p className="text-xs text-muted-foreground mt-4">
              where n is the number of samples and d is the number of features
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-green-600 dark:text-green-400">
              Advantages
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {metadata.pros?.map((pro, idx) => (
                <li key={idx} className="text-sm text-muted-foreground flex items-start gap-2">
                  <span className="text-green-500 mt-1">✓</span>
                  <span>{pro}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-red-600 dark:text-red-400">
              Limitations
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {metadata.cons?.map((con, idx) => (
                <li key={idx} className="text-sm text-muted-foreground flex items-start gap-2">
                  <span className="text-red-500 mt-1">✗</span>
                  <span>{con}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Use Cases</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="grid md:grid-cols-2 gap-3">
            {metadata.use_cases?.map((useCase, idx) => (
              <li
                key={idx}
                className="text-sm text-muted-foreground flex items-center gap-2"
              >
                <span className="w-2 h-2 bg-primary rounded-full" />
                {useCase}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

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
                  className="px-3 py-1 bg-secondary rounded-full text-sm"
                >
                  {algo
                    .split('-')
                    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
                    .join(' ')}
                </span>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Key Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold text-sm mb-1">max_depth</h4>
              <p className="text-sm text-muted-foreground">
                Controls the maximum depth of the tree. Smaller values prevent overfitting
                but may underfit. Set to None for unlimited depth.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-sm mb-1">min_samples_split</h4>
              <p className="text-sm text-muted-foreground">
                The minimum number of samples required to split an internal node. Higher
                values prevent the model from learning overly specific patterns.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-sm mb-1">min_samples_leaf</h4>
              <p className="text-sm text-muted-foreground">
                The minimum number of samples required at a leaf node. This parameter can
                smooth the model, especially in regression problems.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-sm mb-1">criterion</h4>
              <p className="text-sm text-muted-foreground">
                The function to measure split quality. Gini impurity is faster to compute,
                while entropy (information gain) may produce slightly different trees.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
