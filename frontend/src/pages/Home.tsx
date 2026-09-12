import { Link } from 'react-router-dom';
import {
  Brain,
  Network,
  MessageSquare,
  Eye,
  Gamepad2,
  ArrowRight,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/common/Card';

const categories = [
  {
    id: 'ml',
    name: 'Machine Learning',
    description: 'Classic ML algorithms including regression, classification, and clustering',
    icon: Brain,
    path: '/ml',
    color: 'text-blue-500',
  },
  {
    id: 'deep-learning',
    name: 'Deep Learning',
    description: 'Neural networks, CNNs, RNNs, and advanced deep learning architectures',
    icon: Network,
    path: '/deep-learning',
    color: 'text-purple-500',
  },
  {
    id: 'nlp',
    name: 'Natural Language Processing',
    description: 'Text processing, sentiment analysis, and language models',
    icon: MessageSquare,
    path: '/nlp',
    color: 'text-green-500',
  },
  {
    id: 'computer-vision',
    name: 'Computer Vision',
    description: 'Image classification, object detection, and image segmentation',
    icon: Eye,
    path: '/computer-vision',
    color: 'text-orange-500',
  },
  {
    id: 'reinforcement-learning',
    name: 'Reinforcement Learning',
    description: 'Q-Learning, policy gradients, and multi-armed bandits',
    icon: Gamepad2,
    path: '/reinforcement-learning',
    color: 'text-red-500',
  },
];

function Home() {
  return (
    <div className="space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold tracking-tight">
          AI Algorithms Interactive Demo
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Explore and visualize various AI algorithms in action. Select a category below
          to get started.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {categories.map((category) => {
          const Icon = category.icon;
          return (
            <Link key={category.id} to={category.path}>
              <Card className="h-full transition-all hover:shadow-lg hover:scale-[1.02]">
                <CardHeader>
                  <div className="flex items-center space-x-3">
                    <Icon className={`h-8 w-8 ${category.color}`} />
                    <CardTitle className="text-xl">{category.name}</CardTitle>
                  </div>
                  <CardDescription className="mt-2">
                    {category.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center text-sm text-primary font-medium">
                    Explore algorithms
                    <ArrowRight className="ml-1 h-4 w-4" />
                  </div>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>

      <div className="mt-12 rounded-lg bg-muted p-8 text-center">
        <h2 className="text-2xl font-semibold mb-2">Getting Started</h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Each category contains interactive demos of popular algorithms. You can adjust
          parameters, visualize the results, and understand how different algorithms work
          under various conditions.
        </p>
      </div>
    </div>
  );
}

export default Home;
