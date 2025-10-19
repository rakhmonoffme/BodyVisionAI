'use client';

import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BodyComposition } from '@/types';

interface BodyCompositionChartProps {
  bodyComposition: BodyComposition;
}

export function BodyCompositionChart({ bodyComposition }: BodyCompositionChartProps) {
  const data = [
    { name: 'Lean Mass', value: bodyComposition.lean_mass || 0, color: '#3B82F6' },
    { name: 'Fat Mass', value: bodyComposition.fat_mass || 0, color: '#F59E0B' },
  ];

  const bmi = bodyComposition.bmi || 0;
  let bmiStatus = 'Normal';
  let bmiColor = 'text-green-600';

  if (bmi < 18.5) {
    bmiStatus = 'Underweight';
    bmiColor = 'text-yellow-600';
  } else if (bmi >= 25 && bmi < 30) {
    bmiStatus = 'Overweight';
    bmiColor = 'text-orange-600';
  } else if (bmi >= 30) {
    bmiStatus = 'Obese';
    bmiColor = 'text-red-600';
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Body Mass Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-2 gap-4 mt-4">
            <div className="text-center p-3 bg-blue-50 rounded-lg">
              <p className="text-sm text-gray-600">Lean Mass</p>
              <p className="text-2xl font-bold text-blue-600">
                {bodyComposition.lean_mass?.toFixed(1)} kg
              </p>
            </div>
            <div className="text-center p-3 bg-orange-50 rounded-lg">
              <p className="text-sm text-gray-600">Fat Mass</p>
              <p className="text-2xl font-bold text-orange-600">
                {bodyComposition.fat_mass?.toFixed(1)} kg
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Health Metrics</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="p-4 bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg">
            <p className="text-sm text-gray-600">Body Fat Percentage</p>
            <p className="text-3xl font-bold text-blue-600 mt-1">
              {bodyComposition.body_fat_percentage?.toFixed(1)}%
            </p>
          </div>

          <div className="p-4 bg-gradient-to-r from-green-50 to-green-100 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Body Mass Index</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{bmi.toFixed(1)}</p>
              </div>
              <div className={`text-right ${bmiColor}`}>
                <p className="text-sm font-semibold">{bmiStatus}</p>
                <p className="text-xs mt-1">18.5-24.9 normal</p>
              </div>
            </div>
          </div>

          <div className="p-4 bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg">
            <p className="text-sm text-gray-600">Body Type</p>
            <p className="text-2xl font-bold text-purple-600 mt-1 capitalize">
              {bodyComposition.body_type}
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
