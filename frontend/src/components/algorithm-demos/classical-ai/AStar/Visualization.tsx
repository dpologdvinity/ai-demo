interface VisualizationProps {
  grid: number[][];
  start: [number, number];
  goal: [number, number];
  path: number[][];
  pathFound: boolean;
  highlightedNode?: [number, number];
}

export default function Visualization({
  grid,
  start,
  goal,
  path,
  pathFound,
  highlightedNode,
}: VisualizationProps) {
  const gridSize = grid.length;

  const getCellColor = (row: number, col: number) => {
    // Obstacles
    if (grid[row][col] === 1) {
      return 'bg-slate-700';
    }

    // Start
    if (row === start[0] && col === start[1]) {
      return 'bg-neon-green';
    }

    // Goal
    if (row === goal[0] && col === goal[1]) {
      return 'bg-neon-magenta';
    }

    // Path
    if (path.some((p) => p[0] === row && p[1] === col)) {
      return 'bg-neon-cyan';
    }

    // Highlighted node (current frontier exploration)
    if (highlightedNode && row === highlightedNode[0] && col === highlightedNode[1]) {
      return 'bg-neon-violet';
    }

    // Free cell
    return 'bg-slate-800 border border-slate-700';
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <div
        className="grid gap-0.5 bg-slate-900 p-2 rounded-lg"
        style={{
          gridTemplateColumns: `repeat(${gridSize}, minmax(20px, 1fr))`,
          width: `${Math.min(gridSize * 24, 400)}px`,
          height: `${Math.min(gridSize * 24, 400)}px`,
        }}
      >
        {grid.map((row, rowIdx) =>
          row.map((_, colIdx) => (
            <div
              key={`${rowIdx}-${colIdx}`}
              className={`${getCellColor(rowIdx, colIdx)} rounded-sm transition-colors`}
            />
          ))
        )}
      </div>

      <div className="text-sm space-y-2">
        <p>
          <span className="inline-block w-4 h-4 bg-neon-green rounded-sm mr-2" />
          Start
        </p>
        <p>
          <span className="inline-block w-4 h-4 bg-neon-magenta rounded-sm mr-2" />
          Goal
        </p>
        <p>
          <span className="inline-block w-4 h-4 bg-neon-cyan rounded-sm mr-2" />
          Path
        </p>
        <p>
          <span className="inline-block w-4 h-4 bg-neon-violet rounded-sm mr-2" />
          Current Exploration
        </p>
        <p>
          <span className="inline-block w-4 h-4 bg-slate-700 rounded-sm mr-2" />
          Obstacle
        </p>
      </div>

      {pathFound && <p className="text-neon-green font-semibold">Path found!</p>}
      {!pathFound && <p className="text-neon-magenta font-semibold">No path exists</p>}
    </div>
  );
}
