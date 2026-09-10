import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About YOLO Detection</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            What is YOLO?
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            YOLO (You Only Look Once) is a real-time object detection algorithm that treats
            detection as a regression problem. Unlike traditional methods that apply classifiers
            to multiple regions, YOLO divides the image into a grid and predicts bounding boxes
            and class probabilities directly in a single forward pass. YOLOv8 uses a modified
            CSPDarknet backbone and anchor-free detection heads for improved accuracy.
          </p>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            How It Works
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Divide image into SxS grid</li>
            <li>For each grid cell, predict bounding box coordinates and objectness score</li>
            <li>Predict class probabilities for detected objects</li>
            <li>Apply Non-Maximum Suppression to remove duplicate detections</li>
            <li>Return final set of bounding boxes with confidence scores</li>
          </ol>
        </section>

        <section>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Key Advantages
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Real-time performance:</strong> 30+ FPS enables video processing
            </li>
            <li>
              <strong>Single unified network:</strong> End-to-end training without separate
              components
            </li>
            <li>
              <strong>Global context:</strong> Sees entire image during training for better
              performance
            </li>
            <li>
              <strong>Generalized representations:</strong> Learns robust object features
            </li>
          </ul>
        </section>
      </CardContent>
    </Card>
  );
}
