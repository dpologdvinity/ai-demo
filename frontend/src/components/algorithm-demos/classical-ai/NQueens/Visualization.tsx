import { Card, CardContent, CardHeader, CardTitle } from '@/components/common/Card';
import { cn } from '@/lib/utils';
import { Crown } from 'lucide-react';

interface VisualizationProps {
  boardSize: number;
  boardState: number[];
  currentColumn?: number;
  currentRowTried?: number;
  accepted?: boolean;
}

function Visualization({
  boardSize,
  boardState,
  currentColumn = -1,
  currentRowTried = -1,
  accepted = false,
}: VisualizationProps) {
  const cells = [];

  for (let col = 0; col < boardSize; col++) {
    for (let row = 0; row < boardSize; row++) {
      const isLightCell = (row + col) % 2 === 0;
      const hasQueen = boardState[col] === row;
      const isCurrentColumn = col === currentColumn;
      const isCurrentCell = isCurrentColumn && row === currentRowTried;

      let borderColor = '';
      let bgColor = isLightCell ? 'bg-card' : 'bg-muted';

      if (isCurrentColumn) {
        borderColor = 'border-2 border-blue-500';
      }

      if (isCurrentCell) {
        if (accepted) {
          bgColor = isLightCell ? 'bg-green-200 dark:bg-green-900/40' : 'bg-green-100 dark:bg-green-900/30';
        } else {
          bgColor = isLightCell ? 'bg-red-200 dark:bg-red-900/40' : 'bg-red-100 dark:bg-red-900/30';
        }
      }

      cells.push(
        <div
          key={`${row}-${col}`}
          className={cn(
            'w-12 h-12 sm:w-16 sm:h-16 flex items-center justify-center font-bold text-xl sm:text-2xl cursor-default',
            bgColor,
            borderColor || 'border border-border'
          )}
        >
          {hasQueen && (
            <Crown className="w-8 h-8 sm:w-10 sm:h-10 text-yellow-600 dark:text-yellow-400" />
          )}
        </div>
      );
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Chessboard</CardTitle>
      </CardHeader>
      <CardContent className="flex justify-center p-4">
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${boardSize}, minmax(0, 1fr))`,
            gap: 0,
            borderCollapse: 'collapse',
            backgroundColor: 'hsl(var(--border))',
            padding: '4px',
            borderRadius: '4px',
          }}
        >
          {cells}
        </div>
      </CardContent>
    </Card>
  );
}

export default Visualization;
