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
    <header className="sticky top-0 z-50 border-b border-border bg-void/80 backdrop-blur-md">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          <Link to="/" className="flex items-center space-x-2 group">
            <Brain className="h-6 w-6 text-primary transition-all group-hover:drop-shadow-[0_0_8px_rgba(76,243,255,0.8)]" />
            <span className="font-display text-lg font-bold tracking-wide text-glow-cyan">
              AI ALGORITHMS
            </span>
          </Link>

          <nav className="hidden md:flex space-x-1">
            {navigationItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={cn(
                    'relative px-3 py-2 text-sm font-medium transition-colors',
                    isActive
                      ? 'text-primary'
                      : 'text-muted-foreground hover:text-foreground'
                  )}
                >
                  {item.name}
                  {isActive && (
                    <span className="absolute inset-x-2 -bottom-[1px] h-[2px] bg-primary shadow-glow-cyan" />
                  )}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>
    </header>
  );
}

export default Header;
