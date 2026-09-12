import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DocumentationProps {
  algorithmInfo?: any;
}

export function Documentation({ algorithmInfo }: DocumentationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About t-SNE</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is t-SNE?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            t-SNE (t-Distributed Stochastic Neighbor Embedding) is a non-linear dimensionality
            reduction technique designed specifically for visualizing high-dimensional data.
            It converts similarities between data points into joint probabilities and minimizes
            the Kullback-Leibler divergence between these probabilities in high-dimensional
            and low-dimensional spaces.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Compute pairwise similarities in high-dimensional space using Gaussian distribution</li>
            <li>Initialize random positions in low-dimensional space</li>
            <li>Compute pairwise similarities in low-dimensional space using t-distribution</li>
            <li>Minimize KL divergence between the two distributions via gradient descent</li>
            <li>Iterate until convergence or maximum iterations reached</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Parameters
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Perplexity:</strong> Balances local vs. global structure. Think of it as
              the number of close neighbors each point should have. Larger datasets typically
              need higher perplexity (15-50 is common).
            </li>
            <li>
              <strong>Learning Rate:</strong> Controls the step size during optimization. Too
              high can cause poor convergence; too low makes training slow (100-1000 typical).
            </li>
            <li>
              <strong>Iterations:</strong> Number of optimization steps. More iterations allow
              better convergence but take longer (minimum 250, often 1000+ for good results).
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Complexity
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Time Complexity:</strong> O(n²) - Quadratic in number of samples
            </li>
            <li>
              <strong>Space Complexity:</strong> O(n²) - Stores pairwise similarities
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Visualizing high-dimensional datasets (images, text embeddings, gene data)</li>
            <li>Discovering clusters and patterns in complex data</li>
            <li>Feature analysis and exploratory data analysis</li>
            <li>Quality assessment of learned representations</li>
            <li>Interactive data exploration and presentation</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Excellent at preserving local neighborhood structure</li>
            <li>Reveals clusters not visible with linear methods like PCA</li>
            <li>Handles non-linear relationships effectively</li>
            <li>Produces visually appealing and interpretable embeddings</li>
            <li>Widely adopted with extensive research backing</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Limitations
          </h3>
          <ul className="list-disc list-inside space-y-1 text-red-700 dark:text-red-300">
            <li>Computationally expensive for large datasets (O(n²) complexity)</li>
            <li>Non-deterministic - different runs produce different results</li>
            <li>Cannot be applied to new data points (no out-of-sample extension)</li>
            <li>Sensitive to hyperparameter choices (perplexity, learning rate)</li>
            <li>May distort global structure to preserve local relationships</li>
            <li>Distances in embedding space are not directly interpretable</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Tips for Best Results
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Try different perplexity values (5-50) to see various scales of structure</li>
            <li>Run multiple times with different random seeds to ensure stability</li>
            <li>Use at least 1000 iterations for clean, well-separated clusters</li>
            <li>Normalize/scale your data before applying t-SNE</li>
            <li>For very large datasets, consider using a subset or approximations</li>
            <li>Compare with PCA to understand what additional structure t-SNE reveals</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Related Algorithms
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li><strong>PCA:</strong> Linear dimensionality reduction, much faster but less flexible</li>
            <li><strong>UMAP:</strong> Similar to t-SNE but faster and preserves global structure better</li>
            <li><strong>MDS:</strong> Classical dimensionality reduction based on distance preservation</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
