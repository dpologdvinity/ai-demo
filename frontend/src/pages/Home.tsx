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
import { cn } from '@/lib/utils';

const categories = [
  {
    id: 'ml',
    name: 'Machine Learning',
    description: 'Classic ML algorithms including regression, classification, and clustering',
    icon: Brain,
    path: '/ml',
    accent: 'cyan',
  },
  {
    id: 'deep-learning',
    name: 'Deep Learning',
    description: 'Neural networks, CNNs, RNNs, and advanced deep learning architectures',
    icon: Network,
    path: '/deep-learning',
    accent: 'violet',
  },
  {
    id: 'nlp',
    name: 'Natural Language Processing',
    description: 'Text processing, sentiment analysis, and language models',
    icon: MessageSquare,
    path: '/nlp',
    accent: 'green',
  },
  {
    id: 'computer-vision',
    name: 'Computer Vision',
    description: 'Image classification, object detection, and image segmentation',
    icon: Eye,
    path: '/computer-vision',
    accent: 'amber',
  },
  {
    id: 'reinforcement-learning',
    name: 'Reinforcement Learning',
    description: 'Q-Learning, policy gradients, and multi-armed bandits',
    icon: Gamepad2,
    path: '/reinforcement-learning',
    accent: 'magenta',
  },
] as const;

const accentStyles: Record<string, { text: string; bar: string; glow: string }> = {
  cyan: { text: 'text-neon-cyan', bar: 'bg-neon-cyan', glow: 'panel-glow-cyan' },
  violet: { text: 'text-neon-violet', bar: 'bg-neon-violet', glow: 'panel-glow-violet' },
  green: { text: 'text-neon-green', bar: 'bg-neon-green', glow: 'panel-glow-green' },
  amber: { text: 'text-neon-amber', bar: 'bg-neon-amber', glow: 'panel-glow-amber' },
  magenta: { text: 'text-neon-magenta', bar: 'bg-neon-magenta', glow: 'panel-glow-magenta' },
};

function Home() {
  return (
    <div className="space-y-12">
      <div className="max-w-2xl space-y-4">
        <h1 className="font-display text-4xl font-bold tracking-wide text-glow-cyan sm:text-5xl">
          Interactive AI Algorithm Lab
        </h1>
        <p className="max-w-[65ch] text-lg text-muted-foreground">
          Run real machine learning, deep learning, NLP, vision, and reinforcement
          learning algorithms in your browser. Adjust the parameters, watch the
          results update, see how each one actually works.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {categories.map((category) => {
          const Icon = category.icon;
          const accent = accentStyles[category.accent];
          return (
            <Link key={category.id} to={category.path}>
              <Card className={cn('h-full overflow-hidden', accent.glow)}>
                <div className={cn('h-[2px] w-full', accent.bar)} />
                <CardHeader>
                  <div className="flex items-center space-x-3">
                    <Icon className={cn('h-7 w-7', accent.text)} />
                    <CardTitle>{category.name}</CardTitle>
                  </div>
                  <CardDescription className="mt-2">
                    {category.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className={cn('flex items-center text-sm font-medium', accent.text)}>
                    Explore algorithms
                    <ArrowRight className="ml-1 h-4 w-4" />
                  </div>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>
    </div>
  );
}

export default Home;
