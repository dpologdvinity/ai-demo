import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Minimax & Alpha-Beta Pruning</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p>
          Minimax is a recursive algorithm for finding the optimal move in two-player, zero-sum games like tic-tac-toe. It evaluates game states by assuming both players play optimally.
        </p>
        <p>
          The algorithm works by building a game tree where the maximizing player (AI) seeks the highest score, and the minimizing player (opponent) seeks the lowest score. Alpha-beta pruning is an optimization that eliminates branches that cannot affect the final decision, dramatically reducing the number of nodes that must be evaluated.
        </p>
        <p>
          In tic-tac-toe, minimax with alpha-beta pruning guarantees an unbeatable AI opponent that either wins or draws, depending on whether the opponent plays optimally.
        </p>
      </CardContent>
    </Card>
  );
}
