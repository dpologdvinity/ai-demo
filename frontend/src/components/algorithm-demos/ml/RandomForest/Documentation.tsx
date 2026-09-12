import React from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export const Documentation: React.FC = () => {
  return (
    <Tabs defaultValue="theory" className="w-full">
      <TabsList className="grid w-full grid-cols-4">
        <TabsTrigger value="theory">Theory</TabsTrigger>
        <TabsTrigger value="complexity">Complexity</TabsTrigger>
        <TabsTrigger value="use-cases">Use Cases</TabsTrigger>
        <TabsTrigger value="pros-cons">Pros & Cons</TabsTrigger>
      </TabsList>

      <TabsContent value="theory">
        <Card>
          <CardHeader>
            <CardTitle>Algorithm Theory</CardTitle>
            <CardDescription>How Random Forest works</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="font-semibold mb-2">Overview</h4>
              <p className="text-sm text-muted-foreground">
                Random Forest is an ensemble learning method that constructs multiple decision
                trees during training and outputs the class that is the mode of the classes
                predicted by individual trees. This approach combines the predictions of several
                base estimators to improve generalizability and robustness over a single estimator.
              </p>
            </div>

            <div>
              <h4 className="font-semibold mb-2">Key Concepts</h4>
              <ul className="list-disc list-inside space-y-2 text-sm text-muted-foreground">
                <li>
                  <strong>Bootstrap Aggregating (Bagging):</strong> Each tree is trained on a
                  random sample of the data with replacement
                </li>
                <li>
                  <strong>Random Feature Selection:</strong> At each split, only a random subset
                  of features is considered
                </li>
                <li>
                  <strong>Voting Mechanism:</strong> For classification, the final prediction is
                  the majority vote from all trees
                </li>
                <li>
                  <strong>Out-of-Bag (OOB) Evaluation:</strong> Samples not used in training a
                  tree can be used for validation
                </li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-2">How It Works</h4>
              <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                <li>Draw N bootstrap samples from the training data</li>
                <li>For each bootstrap sample, grow a decision tree with random feature selection at each split</li>
                <li>Repeat steps 1-2 for the desired number of trees (n_estimators)</li>
                <li>For prediction, aggregate the predictions from all trees (majority voting)</li>
                <li>Calculate feature importance based on how much each feature decreases impurity across all trees</li>
              </ol>
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="complexity">
        <Card>
          <CardHeader>
            <CardTitle>Time & Space Complexity</CardTitle>
            <CardDescription>Performance characteristics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="font-semibold mb-2">Time Complexity</h4>
              <p className="text-sm text-muted-foreground mb-2">
                <strong>Training:</strong> O(n × log(n) × d × k)
              </p>
              <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground ml-4">
                <li>n = number of samples</li>
                <li>d = number of features</li>
                <li>k = number of trees (n_estimators)</li>
                <li>log(n) comes from the depth of each tree</li>
              </ul>
              <p className="text-sm text-muted-foreground mt-2">
                <strong>Prediction:</strong> O(log(n) × k) - Each tree makes a prediction in
                logarithmic time
              </p>
            </div>

            <div>
              <h4 className="font-semibold mb-2">Space Complexity</h4>
              <p className="text-sm text-muted-foreground">
                <strong>O(k × n)</strong> - Need to store k trees, each potentially storing n
                nodes in the worst case
              </p>
            </div>

            <div>
              <h4 className="font-semibold mb-2">Scalability</h4>
              <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                <li>
                  <strong>Parallelizable:</strong> Trees can be trained independently in parallel
                </li>
                <li>
                  <strong>Memory intensive:</strong> Storing many trees requires significant memory
                </li>
                <li>
                  <strong>Fast prediction:</strong> Once trained, predictions are relatively fast
                </li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="use-cases">
        <Card>
          <CardHeader>
            <CardTitle>Real-World Use Cases</CardTitle>
            <CardDescription>Applications of Random Forest</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">Fraud Detection</h4>
                <p className="text-sm text-muted-foreground">
                  Banks and financial institutions use Random Forest to detect fraudulent
                  transactions by analyzing patterns in transaction data, user behavior, and
                  historical fraud cases. The ensemble approach helps reduce false positives.
                </p>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Stock Market Analysis</h4>
                <p className="text-sm text-muted-foreground">
                  Traders use Random Forest to predict stock price movements by analyzing
                  technical indicators, market sentiment, and historical price data. Feature
                  importance helps identify which factors most influence price changes.
                </p>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Healthcare Predictions</h4>
                <p className="text-sm text-muted-foreground">
                  Medical professionals use Random Forest for disease diagnosis, patient risk
                  stratification, and treatment outcome prediction. It handles the complex,
                  non-linear relationships in medical data well.
                </p>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Other Applications</h4>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                  <li>Customer churn prediction</li>
                  <li>Image classification</li>
                  <li>Recommendation systems</li>
                  <li>Credit scoring</li>
                  <li>Environmental monitoring</li>
                  <li>Bioinformatics</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="pros-cons">
        <Card>
          <CardHeader>
            <CardTitle>Advantages & Limitations</CardTitle>
            <CardDescription>When to use Random Forest</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div>
                <h4 className="font-semibold mb-3 text-green-600">Advantages</h4>
                <ul className="space-y-2">
                  <li className="text-sm">
                    <strong>Reduces Overfitting:</strong> By averaging multiple trees, Random
                    Forest is much less prone to overfitting than individual decision trees
                  </li>
                  <li className="text-sm">
                    <strong>Handles Missing Values:</strong> Can maintain accuracy even when a
                    large proportion of data is missing
                  </li>
                  <li className="text-sm">
                    <strong>Feature Importance:</strong> Provides built-in feature importance
                    scores to help understand which features matter most
                  </li>
                  <li className="text-sm">
                    <strong>No Feature Scaling Required:</strong> Works well with both normalized
                    and unnormalized data
                  </li>
                  <li className="text-sm">
                    <strong>Robust to Outliers:</strong> The ensemble approach makes it resilient
                    to outliers and noise in the data
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold mb-3 text-red-600">Limitations</h4>
                <ul className="space-y-2">
                  <li className="text-sm">
                    <strong>Computationally Expensive:</strong> Training many trees can be slow,
                    especially with large datasets
                  </li>
                  <li className="text-sm">
                    <strong>Memory Intensive:</strong> Storing multiple trees requires significant
                    memory, which can be problematic for deployment
                  </li>
                  <li className="text-sm">
                    <strong>Less Interpretable:</strong> While individual trees are interpretable,
                    understanding an ensemble of hundreds of trees is difficult
                  </li>
                  <li className="text-sm">
                    <strong>Not Ideal for Small Datasets:</strong> May not provide significant
                    benefit over simpler models when data is limited
                  </li>
                  <li className="text-sm">
                    <strong>Biased with Imbalanced Data:</strong> Can be biased toward the
                    majority class in highly imbalanced datasets
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Related Algorithms</h4>
                <p className="text-sm text-muted-foreground">
                  Consider these alternatives: Decision Trees (simpler), Gradient Boosting
                  (often more accurate), AdaBoost (adaptive boosting), XGBoost (optimized
                  gradient boosting)
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  );
};
