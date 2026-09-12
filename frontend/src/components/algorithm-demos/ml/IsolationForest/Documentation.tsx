import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { CheckCircle2, XCircle, Clock, Database } from 'lucide-react';

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
    <div className="space-y-6">
      {/* Overview */}
      <Card>
        <CardHeader>
          <CardTitle>Overview</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="text-muted-foreground mb-2">{metadata.description}</p>
            <div className="flex flex-wrap gap-2 mt-3">
              <Badge variant="secondary">{metadata.difficulty}</Badge>
              {metadata.tags.map((tag) => (
                <Badge key={tag} variant="outline">
                  {tag}
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* How It Works */}
      <Card>
        <CardHeader>
          <CardTitle>How It Works</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="prose prose-sm max-w-none">
            <p className="whitespace-pre-line text-muted-foreground">
              {metadata.theory}
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Use Cases */}
      <Card>
        <CardHeader>
          <CardTitle>Common Use Cases</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {metadata.use_cases.map((useCase, index) => (
              <li key={index} className="flex items-start gap-2">
                <Database className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                <span className="text-sm text-muted-foreground">{useCase}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* Complexity */}
      <Card>
        <CardHeader>
          <CardTitle>Computational Complexity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <Clock className="h-4 w-4 text-primary" />
              <div>
                <p className="text-sm font-medium">Time Complexity</p>
                <code className="text-xs text-muted-foreground">
                  {metadata.complexity.time}
                </code>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Database className="h-4 w-4 text-primary" />
              <div>
                <p className="text-sm font-medium">Space Complexity</p>
                <code className="text-xs text-muted-foreground">
                  {metadata.complexity.space}
                </code>
              </div>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-4">
            Where n = number of samples, d = number of features
          </p>
        </CardContent>
      </Card>

      {/* Pros and Cons */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green-600" />
              Advantages
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {metadata.pros.map((pro, index) => (
                <li key={index} className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                  <span className="text-sm text-muted-foreground">{pro}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <XCircle className="h-5 w-5 text-red-600" />
              Limitations
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {metadata.cons.map((con, index) => (
                <li key={index} className="flex items-start gap-2">
                  <XCircle className="h-4 w-4 text-red-600 mt-0.5 flex-shrink-0" />
                  <span className="text-sm text-muted-foreground">{con}</span>
                </li>
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
            {metadata.related_algorithms.map((algo) => (
              <Badge key={algo} variant="secondary">
                {algo}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Key Parameters */}
      <Card>
        <CardHeader>
          <CardTitle>Key Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold text-sm mb-1">n_estimators</h4>
              <p className="text-sm text-muted-foreground">
                Number of isolation trees in the forest. More trees generally improve detection
                accuracy but increase training time. Typical range: 50-300.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-sm mb-1">contamination</h4>
              <p className="text-sm text-muted-foreground">
                Expected proportion of outliers in the dataset. This acts as a threshold for
                classifying points as anomalies. Should match your domain knowledge about outlier prevalence.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-sm mb-1">max_samples</h4>
              <p className="text-sm text-muted-foreground">
                Number of samples drawn to train each tree. Using fewer samples can speed up training
                and may improve detection of local anomalies. 'auto' uses min(256, n_samples).
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Best Practices */}
      <Card>
        <CardHeader>
          <CardTitle>Best Practices</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li className="flex items-start gap-2">
              <span className="text-primary">•</span>
              <span>
                Set contamination based on domain knowledge or estimate from the data
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary">•</span>
              <span>
                Use at least 100 trees for stable results (default is 100)
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary">•</span>
              <span>
                Feature scaling is not required but can help in some cases
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary">•</span>
              <span>
                Use anomaly scores for ranking rather than just binary classification
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary">•</span>
              <span>
                Works best when anomalies are few and significantly different from normal points
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary">•</span>
              <span>
                Consider ensemble approaches with other anomaly detectors for critical applications
              </span>
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
