import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ConfusionMatrix } from '@/components/visualizations/ConfusionMatrix';
import { useState } from 'react';

interface TreeNode {
  id: number;
  type: string;
  class?: number;
  class_name?: string;
  samples: number;
  value?: number[];
  impurity: number;
  feature?: number;
  feature_name?: string;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
}

interface VisualizationProps {
  result: {
    visualization_data: {
      confusion_matrix: number[][];
      labels: string[];
      tree_structure: TreeNode;
      tree_text: string;
      feature_importance: {
        features: string[];
        importance: number[];
      };
    };
  };
}

export function Visualization({ result }: VisualizationProps) {
  const [viewMode, setViewMode] = useState<'confusion' | 'tree'>('confusion');

  const renderTreeNode = (node: TreeNode, depth: number = 0): JSX.Element => {
    const isLeaf = node.type === 'leaf';
    const indent = depth * 20;

    return (
      <div key={node.id} style={{ marginLeft: `${indent}px` }} className="my-2">
        <div
          className={`p-3 rounded-lg border-2 inline-block ${
            isLeaf
              ? 'bg-green-50 dark:bg-green-950 border-green-500'
              : 'bg-blue-50 dark:bg-blue-950 border-blue-500'
          }`}
        >
          {isLeaf ? (
            <div className="text-sm">
              <div className="font-bold text-green-700 dark:text-green-300">
                Class: {node.class_name}
              </div>
              <div className="text-xs text-muted-foreground">
                Samples: {node.samples}
              </div>
              <div className="text-xs text-muted-foreground">
                Impurity: {node.impurity.toFixed(3)}
              </div>
            </div>
          ) : (
            <div className="text-sm">
              <div className="font-bold text-blue-700 dark:text-blue-300">
                {node.feature_name}
              </div>
              <div className="text-xs text-muted-foreground">
                &le; {node.threshold?.toFixed(3)}
              </div>
              <div className="text-xs text-muted-foreground">
                Samples: {node.samples}
              </div>
              <div className="text-xs text-muted-foreground">
                Impurity: {node.impurity.toFixed(3)}
              </div>
            </div>
          )}
        </div>

        {!isLeaf && node.left && (
          <div className="ml-4">
            <div className="text-xs text-muted-foreground ml-2">True:</div>
            {renderTreeNode(node.left, depth + 1)}
          </div>
        )}
        {!isLeaf && node.right && (
          <div className="ml-4">
            <div className="text-xs text-muted-foreground ml-2">False:</div>
            {renderTreeNode(node.right, depth + 1)}
          </div>
        )}
      </div>
    );
  };

  const renderTreeVisualization = () => {
    // Limit tree depth for visualization
    const maxDepthToRender = 3;

    const limitDepth = (node: TreeNode, depth: number): TreeNode | null => {
      if (depth >= maxDepthToRender || node.type === 'leaf') {
        return node;
      }

      return {
        ...node,
        left: node.left ? (limitDepth(node.left, depth + 1) || undefined) : undefined,
        right: node.right ? (limitDepth(node.right, depth + 1) || undefined) : undefined,
      };
    };

    const limitedTree = limitDepth(result.visualization_data.tree_structure, 0);

    return (
      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          {limitedTree && renderTreeNode(limitedTree)}
          <div className="mt-4 text-xs text-muted-foreground">
            Showing first 3 levels of the tree for clarity
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setViewMode('confusion')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            viewMode === 'confusion'
              ? 'bg-primary text-primary-foreground'
              : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
          }`}
        >
          Confusion Matrix
        </button>
        <button
          onClick={() => setViewMode('tree')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            viewMode === 'tree'
              ? 'bg-primary text-primary-foreground'
              : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
          }`}
        >
          Tree Structure
        </button>
      </div>

      {viewMode === 'confusion' ? (
        <ConfusionMatrix
          matrix={result.visualization_data.confusion_matrix}
          labels={result.visualization_data.labels}
          title="Classification Results"
        />
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Decision Tree Structure</CardTitle>
          </CardHeader>
          <CardContent>
            {renderTreeVisualization()}

            <div className="mt-6 pt-6 border-t">
              <h4 className="font-semibold mb-2">Tree Rules (Text Format)</h4>
              <pre className="text-xs bg-secondary p-4 rounded-lg overflow-x-auto max-h-96">
                {result.visualization_data.tree_text}
              </pre>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
