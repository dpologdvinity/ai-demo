import { Card, CardContent, CardHeader, CardTitle } from '@/components/common/Card';
import { ScatterPlot } from '@/components/visualizations/ScatterPlot';
import { LineChart } from '@/components/visualizations/LineChart';

interface VisualizationProps {
  generationData: Array<{
    generation: number;
    best_fitness: number;
    avg_fitness: number;
    population_sample: number[];
  }>;
  currentGeneration: number;
}

function Visualization({ generationData, currentGeneration }: VisualizationProps) {
  // Generate the fitness landscape curve (f(x) = x*sin(10*pi*x) + 1 over [0,2])
  const landscapePoints = [];
  for (let i = 0; i < 200; i++) {
    const x = (i / 199) * 2;
    const y = x * Math.sin(10 * Math.PI * x) + 1;
    landscapePoints.push({ x, fitness: y, type: 'landscape' });
  }

  // Get current generation population sample
  const currentGen = generationData[currentGeneration] || generationData[0];
  const populationPoints = (currentGen?.population_sample || []).map((x) => ({
    x,
    fitness: x * Math.sin(10 * Math.PI * x) + 1,
    type: 'population',
  }));

  // Prepare fitness progression data
  const fitnessProgressionData = generationData.map((g) => ({
    generation: g.generation,
    best_fitness: g.best_fitness,
    avg_fitness: g.avg_fitness,
  }));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Population on Fitness Landscape</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96">
            <ScatterPlot
              data={[landscapePoints, populationPoints]}
              xKey="x"
              yKey="fitness"
              xLabel="x (Solution Space)"
              yLabel="Fitness"
              height={350}
              showGrid
              showLegend
              clusterNames={['Landscape', `Generation ${currentGen?.generation || 0}`]}
              colors={['#999', '#3b82f6']}
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Fitness Progression</CardTitle>
        </CardHeader>
        <CardContent>
          <LineChart
            data={fitnessProgressionData}
            xKey="generation"
            yKey={['best_fitness', 'avg_fitness']}
            xLabel="Generation"
            yLabel="Fitness"
            height={250}
            showGrid
            showLegend
          />
        </CardContent>
      </Card>
    </div>
  );
}

export default Visualization;
