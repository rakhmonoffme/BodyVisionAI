'use client';

import { LucideIcon } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface MeasurementCardProps {
  icon: LucideIcon;
  label: string;
  value: number;
  unit?: string;
  referenceMin?: number | null;
  referenceMax?: number | null;
}

export function MeasurementCard({
  icon: Icon,
  label,
  value,
  unit = 'cm',
  referenceMin,
  referenceMax,
}: MeasurementCardProps) {
  const isInRange =
    referenceMin !== null &&
    referenceMin !== undefined &&
    referenceMax !== null &&
    referenceMax !== undefined &&
    value >= referenceMin &&
    value <= referenceMax;

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
              <Icon className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900 capitalize">{label}</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">
                {value.toFixed(1)} <span className="text-base text-gray-600">{unit}</span>
              </p>
            </div>
          </div>
          {referenceMin !== null && referenceMax !== null && (
            <Badge variant={isInRange ? 'default' : 'secondary'} className="mt-1">
              {isInRange ? 'Normal' : 'Check'}
            </Badge>
          )}
        </div>
        {referenceMin !== null && referenceMax !== null && (
          <p className="text-xs text-gray-500 mt-3">
            Normal range: {referenceMin}-{referenceMax} {unit}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
