import { useQuery } from '@tanstack/react-query';
import { Network } from 'lucide-react';
import { apiService } from '@/services/api';
import { AlgorithmCategory } from '@/types';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/common/Card';
import Button from '@/components/common/Button';

function DeepLearning() {
  const { data: algorithms, isLoading, error } = useQuery({
    queryKey: ['algorithms', AlgorithmCategory.DeepLearning],
    queryFn: () => apiService.getAlgorithms(AlgorithmCategory.DeepLearning),
  });

  return (
    <div className="space-y-8">
      <div className="flex items-center space-x-4">
        <Network className="h-10 w-10 text-purple-500" />
        <div>
          <h1 className="text-3xl font-bold">Deep Learning</h1>
          <p className="text-muted-foreground">
            Neural networks and advanced deep learning architectures
          </p>
        </div>
      </div>

      {isLoading && (
        <div className="text-center py-12">
          <p className="text-muted-foreground">Loading algorithms...</p>
        </div>
      )}

      {error && (
        <div className="text-center py-12">
          <p className="text-destructive">Failed to load algorithms. Please try again.</p>
        </div>
      )}

      {algorithms && algorithms.length === 0 && (
        <div className="text-center py-12">
          <p className="text-muted-foreground">No algorithms available yet.</p>
        </div>
      )}

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {algorithms?.map((algorithm) => (
          <Card key={algorithm.id}>
            <CardHeader>
              <CardTitle>{algorithm.name}</CardTitle>
              <CardDescription>{algorithm.description}</CardDescription>
            </CardHeader>
            <CardContent>
              {algorithm.complexity && (
                <div className="text-sm space-y-1 mb-4">
                  <p className="text-muted-foreground">
                    <span className="font-medium">Time:</span> {algorithm.complexity.time}
                  </p>
                  <p className="text-muted-foreground">
                    <span className="font-medium">Space:</span> {algorithm.complexity.space}
                  </p>
                </div>
              )}
              <Button className="w-full">Try Algorithm</Button>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

export default DeepLearning;
