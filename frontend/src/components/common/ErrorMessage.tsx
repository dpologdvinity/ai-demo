import { AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ErrorMessageProps {
  message: string;
  className?: string;
}

function ErrorMessage({ message, className }: ErrorMessageProps) {
  return (
    <div
      className={cn(
        'flex items-center gap-2 rounded-lg border border-destructive bg-destructive/10 p-4 text-destructive',
        className
      )}
      role="alert"
    >
      <AlertCircle className="h-5 w-5 flex-shrink-0" />
      <p className="text-sm font-medium">{message}</p>
    </div>
  );
}

export default ErrorMessage;
