import { Link, useLocation } from 'react-router-dom';
import { Brain } from 'lucide-react';
import { cn } from '@/lib/utils';

const navigationItems = [
  { name: 'Home', path: '/' },
  { name: 'Machine Learning', path: '/ml' },
  { name: 'Deep Learning', path: '/deep-learning' },
  { name: 'NLP', path: '/nlp' },
  { name: 'Computer Vision', path: '/computer-vision' },
  { name: 'Reinforcement Learning', path: '/reinforcement-learning' },
];

function Header() {
  const location = useLocation();

  return (
    <header className="border-b bg-card">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          <Link to="/" className="flex items-center space-x-2">
            <Brain className="h-6 w-6 text-primary" />
            <span className="text-xl font-bold">AI Algorithms Demo</span>
          </Link>

          <nav className="hidden md:flex space-x-1">
            {navigationItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={cn(
                  'px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  location.pathname === item.path
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                )}
              >
                {item.name}
              </Link>
            ))}
          </nav>
        </div>
      </div>
    </header>
  );
}

export default Header;
