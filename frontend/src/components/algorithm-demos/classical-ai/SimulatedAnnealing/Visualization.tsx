import { Card, CardContent, CardHeader, CardTitle } from '@/components/common/Card';
import { ScatterPlot } from '@/components/visualizations/ScatterPlot';
import { LineChart } from '@/components/visualizations/LineChart';

interface StepData {
  step: number;
  temperature: number;
  current_solution: number;
  current_cost: number;
  candidate_solution: number;
  candidate_cost: number;
  accepted: boolean;
}

interface VisualizationProps {
  stepData: StepData[];
  currentStep: number;
}

function Visualization({ stepData, currentStep }: VisualizationProps) {
  const currentTrace = stepData[currentStep] || stepData[0];

  // Generate cost landscape (cost = -(x*sin(10*pi*x)+1))
  const landscapePoints = [];
  for (let i = 0; i < 200; i++) {
    const x = (i / 199) * 2;
    const cost = -(x * Math.sin(10 * Math.PI * x) + 1);
    landscapePoints.push({ x, cost, type: 'landscape' });
  }

  // Current solution point
  const currentPoint = [
    {
      x: currentTrace?.current_solution || 0,
      cost: currentTrace?.current_cost || 0,
      type: 'current',
    },
  ];

  // Prepare progression data
  const progressionData = stepData.map((s) => ({
    step: s.step,
    temperature: s.temperature,
    current_cost: s.current_cost,
  }));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Solution on Cost Landscape</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96">
            <ScatterPlot
              data={[landscapePoints, currentPoint]}
              xKey="x"
              yKey="cost"
              xLabel="x (Solution Space)"
              yLabel="Cost"
              height={350}
              showGrid
              showLegend
              clusterNames={['Landscape', `Step ${currentTrace?.step || 0}`]}
              colors={['#999', '#ef4444']}
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Temperature & Cost Progression</CardTitle>
        </CardHeader>
        <CardContent>
          <LineChart
            data={progressionData}
            xKey="step"
            yKey={['temperature', 'current_cost']}
            xLabel="Iteration"
            yLabel="Temperature / Cost"
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
