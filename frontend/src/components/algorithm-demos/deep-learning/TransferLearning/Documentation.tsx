import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Transfer Learning</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is Transfer Learning?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            Transfer learning is a machine learning technique where a model trained on one task is
            reused as a starting point for training on a different (but related) task. Instead of training
            a model from scratch, we leverage the learned features from a pre-trained model on a large dataset
            (like ImageNet) and adapt it to our specific problem. This approach dramatically reduces training
            time, data requirements, and computational resources while often achieving better accuracy.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Load a pre-trained model (e.g., ResNet trained on ImageNet)</li>
            <li>Freeze early layers containing general features (edges, textures, shapes)</li>
            <li>Replace the final classification layer with new output classes for your task</li>
            <li>Fine-tune the model by training only unfrozen layers on your dataset</li>
            <li>Use a lower learning rate to make small adjustments to pre-learned features</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Transfer Learning Strategies
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Feature Extraction:</strong> Freeze all but the final layer, train only new classifier.
              Best when: target dataset is small, tasks are very similar.
            </li>
            <li>
              <strong>Fine-tuning:</strong> Gradually unfreeze and fine-tune later layers. Best for medium datasets
              with moderately different tasks.
            </li>
            <li>
              <strong>Full Training:</strong> Train all layers (but with low learning rate). Best when target
              dataset is large and tasks differ significantly.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Advantages
          </h3>
          <ul className="list-disc list-inside space-y-1 text-green-700 dark:text-green-300">
            <li>Much faster training (hours instead of days/weeks)</li>
            <li>Requires far less labeled data (often 10-100x less)</li>
            <li>Better generalization due to pre-learned features</li>
            <li>Lower computational requirements</li>
            <li>Excellent results even with limited data</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            When to Use Each Strategy
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Feature Extraction:</strong> Small dataset (1K-10K images), similar domain to ImageNet
            </li>
            <li>
              <strong>Fine-tuning:</strong> Medium dataset (10K-100K), moderately different domain
            </li>
            <li>
              <strong>Full Training:</strong> Large dataset (100K+), significantly different domain
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Parameters
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Learning Rate:</strong> Use lower values (0.0001-0.001) for transfer learning
              vs. training from scratch (0.01+). Lower rates prevent destroying learned features.
            </li>
            <li>
              <strong>Batch Size:</strong> Smaller batches (8-32) work well with transfer learning
              and limited data.
            </li>
            <li>
              <strong>Epochs:</strong> Usually converges faster (5-20 epochs) than training from scratch.
            </li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Use Cases
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Medical image analysis (limited labeled data)</li>
            <li>Autonomous driving perception systems</li>
            <li>Plant disease detection</li>
            <li>Custom object detection</li>
            <li>Domain adaptation (source vs. target domain)</li>
          </ul>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Tips for Success
          </h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Use models pre-trained on ImageNet as default</li>
            <li>Start with feature extraction strategy, then try fine-tuning if needed</li>
            <li>Use data augmentation to extend limited training data</li>
            <li>Monitor both training and validation metrics for overfitting</li>
            <li>Ensure input images are normalized the same way as training data</li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
