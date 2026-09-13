import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';

function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Simulated Annealing</CardTitle>
        <CardDescription>Probabilistic Metaheuristic for Global Optimization</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <p>
          Simulated annealing is a probabilistic technique inspired by metallurgy, where materials are heated and then slowly cooled to achieve a low-energy crystalline state. The algorithm starts with a high temperature that allows the search to explore widely and occasionally accept worse solutions, gradually cooling to exploit promising regions.
        </p>
        <p>
          At each iteration, the algorithm evaluates a nearby candidate solution. If it improves the cost, it's always accepted. If it's worse, it may still be accepted with probability e^(-ΔE/T), where ΔE is the cost difference and T is the current temperature. This acceptance of "uphill" moves helps escape local optima. As temperature decreases, the acceptance probability becomes stricter, transitioning from exploration to exploitation.
        </p>
        <p>
          The effectiveness of simulated annealing depends on the balance between the initial temperature (higher allows more exploration) and the cooling rate (slower cooling allows better convergence). It's particularly useful for hard combinatorial optimization problems where the solution space is irregular with many local optima.
        </p>
      </CardContent>
    </Card>
  );
}

export default Documentation;
