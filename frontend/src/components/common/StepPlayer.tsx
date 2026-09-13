import { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card } from '@/components/common/Card';
import { cn } from '@/lib/utils';

export interface StepPlayerProps {
  totalSteps: number;
  currentStep: number;
  onStepChange: (step: number) => void;
  isPlaying: boolean;
  onPlayingChange: (playing: boolean) => void;
  speed: number; // multiplier, e.g. 0.5, 1, 2, 4
  onSpeedChange: (speed: number) => void;
  className?: string;
}

/**
 * StepPlayer: A playback control bar for animating through algorithm steps.
 * Pure controlled component -- parent owns the interval/timer via useStepPlayback hook.
 */
export function StepPlayer({
  totalSteps,
  currentStep,
  onStepChange,
  isPlaying,
  onPlayingChange,
  speed,
  onSpeedChange,
  className,
}: StepPlayerProps) {
  const speedOptions = [0.5, 1, 2, 4];

  return (
    <Card className={cn('p-3', className)}>
      <div className="flex items-center gap-3">
        {/* Play/Pause button */}
        <Button
          size="icon"
          variant="ghost"
          onClick={() => onPlayingChange(!isPlaying)}
          className="hover:text-neon-cyan"
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? (
            <Pause className="h-5 w-5" />
          ) : (
            <Play className="h-5 w-5" />
          )}
        </Button>

        {/* Reset button */}
        <Button
          size="icon"
          variant="ghost"
          onClick={() => {
            onStepChange(0);
            onPlayingChange(false);
          }}
          className="hover:text-neon-cyan"
          aria-label="Reset to step 0"
        >
          <RotateCcw className="h-5 w-5" />
        </Button>

        {/* Scrubber slider */}
        <div className="flex-1 min-w-[150px]">
          <Slider
            min={0}
            max={Math.max(0, totalSteps - 1)}
            step={1}
            value={[currentStep]}
            onValueChange={(value) => {
              onStepChange(value[0]);
              onPlayingChange(false); // pause on manual scrub
            }}
          />
        </div>

        {/* Step counter */}
        <div className="font-mono text-sm whitespace-nowrap">
          Step {currentStep + 1} / {totalSteps}
        </div>

        {/* Speed selector */}
        <div className="flex gap-2">
          {speedOptions.map((s) => (
            <Button
              key={s}
              size="sm"
              variant={speed === s ? 'default' : 'outline'}
              onClick={() => onSpeedChange(s)}
              className={cn(
                'px-2 text-xs',
                speed === s && 'text-neon-cyan border-neon-cyan shadow-glow-cyan'
              )}
            >
              {s}x
            </Button>
          ))}
        </div>
      </div>
    </Card>
  );
}

/**
 * useStepPlayback: Custom hook that manages currentStep/isPlaying/speed state
 * and the setInterval loop advancing currentStep while isPlaying.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function useStepPlayback(totalSteps: number) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  // Main playback loop
  useEffect(() => {
    if (!isPlaying || totalSteps <= 1) return;

    const intervalDuration = 600 / speed;
    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= totalSteps - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, intervalDuration);

    return () => clearInterval(interval);
  }, [isPlaying, speed, totalSteps]);

  // Ensure currentStep never exceeds bounds
  useEffect(() => {
    if (currentStep >= totalSteps) {
      setCurrentStep(Math.max(0, totalSteps - 1));
    }
  }, [totalSteps, currentStep]);

  const playerProps = {
    currentStep,
    onStepChange: setCurrentStep,
    isPlaying,
    onPlayingChange: setIsPlaying,
    speed,
    onSpeedChange: setSpeed,
  };

  return {
    currentStep,
    isPlaying,
    speed,
    setCurrentStep,
    setIsPlaying,
    setSpeed,
    playerProps,
  };
}
