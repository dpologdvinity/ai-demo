import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About A* Pathfinding</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p>
          A* is an informed search algorithm that finds the shortest path from a start node to a goal node in a weighted graph. It combines the benefits of Dijkstra's algorithm with heuristic guidance.
        </p>
        <p>
          The algorithm evaluates nodes based on f(n) = g(n) + h(n), where g(n) is the actual cost from the start, and h(n) is the estimated cost to the goal. By always expanding the node with the lowest f-value, A* guarantees finding the optimal path while maintaining efficiency through the heuristic estimate.
        </p>
        <p>
          Different heuristics (Manhattan, Euclidean, Chebyshev) trade off between accuracy and computation. Manhattan distance is commonly used on grids where only orthogonal movement is allowed.
        </p>
      </CardContent>
    </Card>
  );
}
