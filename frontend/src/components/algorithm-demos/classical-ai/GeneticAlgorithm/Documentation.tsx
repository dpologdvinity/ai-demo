import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/common/Card';

function Documentation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About Genetic Algorithms</CardTitle>
        <CardDescription>Population-based Evolutionary Optimization</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <p>
          Genetic algorithms are evolutionary algorithms inspired by natural selection and biological genetics. They work by maintaining a population of candidate solutions that evolve over generations through mechanisms like selection, crossover (recombination), and mutation.
        </p>
        <p>
          In this demo, we optimize the function f(x) = x*sin(10πx) + 1 over the interval [0,2], which has multiple local maxima. Each individual in the population represents a candidate solution (an x value). At each generation, individuals are selected based on fitness, pairs are crossed over to produce offspring, and random mutations are applied to maintain diversity. Fitter individuals are more likely to be selected, allowing the population to converge toward global optima over generations.
        </p>
        <p>
          The balance between <strong>exploration</strong> (mutation, diversity) and <strong>exploitation</strong> (selection, convergence) determines the algorithm's effectiveness. This classic approach demonstrates how simple rules applied iteratively can discover good solutions without explicit knowledge of the problem structure.
        </p>
      </CardContent>
    </Card>
  );
}

export default Documentation;
