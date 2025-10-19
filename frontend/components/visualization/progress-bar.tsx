'use client';

import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';

interface ProgressBarProps {
  progress: number;
  currentStep: string;
  className?: string;
}

const stages = [
  { label: 'Validating photos', minProgress: 0, maxProgress: 20 },
  { label: 'Extracting measurements', minProgress: 20, maxProgress: 50 },
  { label: 'Generating 3D model', minProgress: 50, maxProgress: 80 },
  { label: 'Calculating body composition', minProgress: 80, maxProgress: 100 },
];

export function ProgressBar({ progress, currentStep, className }: ProgressBarProps) {
  const currentStageIndex = stages.findIndex(
    (stage) => progress >= stage.minProgress && progress <= stage.maxProgress
  );

  return (
    <div className={cn('space-y-6', className)}>
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium text-gray-900">{currentStep}</span>
          <span className="text-gray-600">{progress}%</span>
        </div>
        <Progress value={progress} className="h-3" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {stages.map((stage, index) => {
          const isCompleted = progress > stage.maxProgress;
          const isCurrent = index === currentStageIndex;
          const isPending = progress < stage.minProgress;

          return (
            <div
              key={stage.label}
              className={cn(
                'flex items-center gap-3 p-3 rounded-lg border transition-all',
                isCompleted && 'bg-green-50 border-green-200',
                isCurrent && 'bg-blue-50 border-blue-300 shadow-sm',
                isPending && 'bg-gray-50 border-gray-200'
              )}
            >
              <div
                className={cn(
                  'w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold',
                  isCompleted && 'bg-green-500 text-white',
                  isCurrent && 'bg-blue-500 text-white',
                  isPending && 'bg-gray-300 text-gray-600'
                )}
              >
                {index + 1}
              </div>
              <span
                className={cn(
                  'text-sm font-medium',
                  isCompleted && 'text-green-700',
                  isCurrent && 'text-blue-700',
                  isPending && 'text-gray-500'
                )}
              >
                {stage.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
