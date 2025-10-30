'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MeasurementCard } from '@/components/visualization/measurement-card';
import { BodyCompositionChart } from '@/components/visualization/body-composition-chart';
import { getResults, getMeasurementImageUrl, getMeshUrl } from '@/lib/api';
import { AnalysisResult } from '@/types';
import { MeshViewer } from '@/components/visualization/mesh_viewer';
import {
  Ruler,
  User2,
  Shirt,
  Heart,
  ChevronDown,
  Activity,
  ArrowLeft,
  Download,
  Loader2,
} from 'lucide-react';

const measurementIcons: Record<string, any> = {
  neck: User2,
  shoulder_width: Shirt,
  chest: Heart,
  waist: Activity,
  abdomen: Activity,
  hip: Activity,
  thigh: Ruler,
  calf: Ruler,
  knee: Ruler,
  ankle: Ruler,
};

const measurementLabels: Record<string, string> = {
  neck: 'Neck',
  shoulder_width: 'Shoulder Width',
  chest: 'Chest',
  waist: 'Waist',
  abdomen: 'Abdomen',
  hip: 'Hip',
  thigh: 'Thigh',
  calf: 'Calf',
  knee: 'Knee',
  ankle: 'Ankle',
};

export default function ResultsPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.sessionId as string;
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!sessionId) {
      router.push('/');
      return;
    }

    const loadResults = async () => {
      try {
        const data = await getResults(sessionId);
        if (data) {
          setResult(data);
        } else {
          router.push('/analyze');
        }
      } catch (error) {
        console.error('Failed to load results:', error);
      } finally {
        setLoading(false);
      }
    };

    loadResults();
  }, [sessionId, router]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading your results...</p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white flex items-center justify-center">
        <Card className="max-w-md">
          <CardContent className="p-8 text-center">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Results Not Found</h2>
            <p className="text-gray-600 mb-6">
              We couldn't find the analysis results.
            </p>
            <Link href="/">
              <Button>Return to Home</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <nav className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link href="/" className="flex items-center gap-2">
              <ArrowLeft className="w-5 h-5" />
              <span className="font-semibold">Back to Home</span>
            </Link>
            <div className="flex items-center gap-2">
              <Activity className="w-6 h-6 text-blue-600" />
              <span className="font-bold text-gray-900">BodyAnalyzer</span>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Your Body Analysis Results</h1>
          <p className="text-gray-600">
            Analysis completed on {new Date(result.timestamp).toLocaleDateString()}
          </p>
        </div>

        <Tabs defaultValue="composition" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 h-auto">
            <TabsTrigger value="composition" className="py-3">
              Body Composition
            </TabsTrigger>
            <TabsTrigger value="measurements" className="py-3">
              Measurements
            </TabsTrigger>
            <TabsTrigger value="3d-model" className="py-3">
              3D Model
            </TabsTrigger>
            <TabsTrigger value="images" className="py-3">
              Analysis Images
            </TabsTrigger>
          </TabsList>

          <TabsContent value="composition" className="space-y-6">
            <Card className="bg-gradient-to-r from-blue-500 to-blue-600 text-white">
              <CardContent className="p-8">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                  <div>
                    <p className="text-blue-100 text-sm mb-1">Body Fat</p>
                    <p className="text-4xl font-bold">{result.bodyComposition.body_fat_percentage}%</p>
                  </div>
                  <div>
                    <p className="text-blue-100 text-sm mb-1">BMI</p>
                    <p className="text-4xl font-bold">{result.bodyComposition.bmi}</p>
                  </div>
                  <div>
                    <p className="text-blue-100 text-sm mb-1">Fat Mass</p>
                    <p className="text-4xl font-bold">{result.bodyComposition.fat_mass_kg} kg</p>
                  </div>
                  <div>
                    <p className="text-blue-100 text-sm mb-1">Lean Mass</p>
                    <p className="text-4xl font-bold">{result.bodyComposition.lean_mass_kg} kg</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Health Assessment</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <p className="text-sm text-gray-600">Category</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {result.healthMetrics.health_status.category}
                    </p>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">Risk Level</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {result.healthMetrics.health_status.risk_level}
                    </p>
                  </div>
                  <div className="p-4 bg-green-50 rounded-lg">
                    <p className="text-sm text-gray-600 mb-2">Recommendation</p>
                    <p className="text-gray-900">{result.healthMetrics.health_status.recommendation}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="measurements" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Ruler className="w-5 h-5" />
                  Body Measurements
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(result.measurements).map(([key, value]) => {
                    const Icon = measurementIcons[key] || Ruler;
                    const label = measurementLabels[key] || key;
                    return (
                      <MeasurementCard
                        key={key}
                        icon={Icon}
                        label={label}
                        value={value}
                        referenceMin={null}
                        referenceMax={null}
                      />
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="3d-model" className="space-y-6">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Interactive 3D Body Model</CardTitle>
                  <a 
                    href={getMeshUrl(sessionId)} 
                    download={`body_mesh_${sessionId}.obj`}
                  >
                    <Button variant="outline" size="sm">
                      <Download className="w-4 h-4 mr-2" />
                      Download Mesh
                    </Button>
                  </a>
                </div>
              </CardHeader>
              <CardContent>
                <MeshViewer sessionId={sessionId} />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="images" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Measurement Visualization</CardTitle>
                <p className="text-sm text-gray-600">
                  See how body measurements were extracted with landmark points
                </p>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {['front', 'side', 'back'].map((view) => (
                    <div key={view}>
                      <p className="text-sm font-medium text-gray-700 mb-2 capitalize">
                        {view} View
                      </p>
                      <div className="border rounded-lg overflow-hidden bg-gray-50">
                        <img
                          src={getMeasurementImageUrl(sessionId, view as 'front' | 'side' | 'back')}
                          alt={`${view} view with measurements`}
                          className="w-full h-auto"
                          onError={(e) => {
                            const target = e.currentTarget;
                            target.style.display = 'none';
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <Card className="mt-8 bg-green-50 border-green-200">
          <CardContent className="p-6">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0">
                <ChevronDown className="w-6 h-6 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">What's Next?</h3>
                <p className="text-gray-700 text-sm mb-4">
                  Your body analysis is complete! Start a new analysis to track your progress.
                </p>
                <Link href="/analyze">
                  <Button size="sm">Start New Analysis</Button>
                </Link>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}