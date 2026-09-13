import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';

function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About N-Queens</CardTitle>
        <CardDescription>Constraint Satisfaction Problem using Backtracking</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <p>
          The N-Queens problem is a classic constraint satisfaction problem where you must place N queens on an N×N chessboard such that no two queens can attack each other. Queens attack along rows, columns, and diagonals, making this a challenging combinatorial optimization problem.
        </p>
        <p>
          This implementation uses <strong>backtracking</strong>, a depth-first search technique that explores possible placements column by column. When a conflict is detected (another queen on the same row or diagonal), the algorithm backtracks and tries a different row. This approach efficiently prunes the search space by abandoning invalid branches early rather than exploring all possible combinations.
        </p>
        <p>
          The solver tracks the number of backtrack operations, showing how many failed attempts were made before finding a valid solution. Larger board sizes exponentially increase the search space, making the backtracking approach essential for tractable solutions.
        </p>
      </CardContent>
    </Card>
  );
}

export default Documentation;
