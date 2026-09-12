import { useQuery } from '@tanstack/react-query';
import { Eye } from 'lucide-react';
import { apiService } from '@/services/api';
import { AlgorithmCategory } from '@/types';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/common/Card';
import Button from '@/components/common/Button';

function ComputerVision() {
  const { data: algorithms, isLoading, error } = useQuery({
    queryKey: ['algorithms', AlgorithmCategory.ComputerVision],
    queryFn: () => apiService.getAlgorithms(AlgorithmCategory.ComputerVision),
  });

  return (
    <div className="space-y-8">
      <div className="flex items-center space-x-4">
        <Eye className="h-10 w-10 text-orange-500" />
        <div>
          <h1 className="text-3xl font-bold">Computer Vision</h1>
          <p className="text-muted-foreground">
            Image classification, object detection, and visual understanding
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

export default ComputerVision;
