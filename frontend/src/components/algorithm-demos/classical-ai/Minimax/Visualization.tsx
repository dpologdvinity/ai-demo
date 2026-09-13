interface VisualizationProps {
  board: string[][];
  winner: 'ai' | 'opponent' | 'draw' | null;
}

export default function Visualization({ board, winner }: VisualizationProps) {
  const getCellColor = (value: string) => {
    if (value === 'X') return 'bg-neon-magenta';
    if (value === 'O') return 'bg-neon-cyan';
    return 'bg-slate-800 border border-slate-700';
  };

  const getWinnerText = () => {
    if (winner === 'ai') return 'AI Wins!';
    if (winner === 'opponent') return 'Opponent Wins!';
    if (winner === 'draw') return "It's a Draw!";
    return '';
  };

  const getWinnerColor = () => {
    if (winner === 'ai') return 'text-neon-magenta';
    if (winner === 'opponent') return 'text-neon-cyan';
    if (winner === 'draw') return 'text-neon-green';
    return '';
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="grid grid-cols-3 gap-2 bg-slate-900 p-4 rounded-lg">
        {board.map((row, rowIdx) =>
          row.map((cell, colIdx) => (
            <div
              key={`${rowIdx}-${colIdx}`}
              className={`${getCellColor(cell)} w-20 h-20 flex items-center justify-center rounded-lg text-2xl font-bold transition-colors`}
            >
              {cell}
            </div>
          ))
        )}
      </div>

      <div className="text-sm space-y-2">
        <p>
          <span className="inline-block w-4 h-4 bg-neon-magenta rounded-sm mr-2" />
          AI (X)
        </p>
        <p>
          <span className="inline-block w-4 h-4 bg-neon-cyan rounded-sm mr-2" />
          Opponent (O)
        </p>
      </div>

      {winner && <p className={`text-xl font-bold ${getWinnerColor()}`}>{getWinnerText()}</p>}
    </div>
  );
}
