'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MeasurementCard } from '@/components/visualization/measurement-card';
import { BodyCompositionChart } from '@/components/visualization/body-composition-chart';
import { getAnalysisResult } from '@/lib/api';
import { AnalysisResult } from '@/types';
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
  shoulders: Shirt,
  chest: Heart,
  waist: Activity,
  abdomen: Activity,
  hips: Activity,
  thighs: Ruler,
  calves: Ruler,
  knees: Ruler,
  ankles: Ruler,
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
        const data = await getAnalysisResult(sessionId);
        if (data && data.session.status === 'completed') {
          setResult(data);
        } else {
          router.push('/processing');
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
              We couldn't find the analysis results. The session may have expired or the analysis
              is still processing.
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
            Analysis completed on {new Date(result.session.created_at).toLocaleDateString()}
          </p>
        </div>

        <Card className="mb-6 bg-gradient-to-r from-blue-500 to-blue-600 text-white">
          <CardContent className="p-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-blue-100 text-sm">Height</p>
                <p className="text-2xl font-bold">{result.session.user_height} cm</p>
              </div>
              <div>
                <p className="text-blue-100 text-sm">Weight</p>
                <p className="text-2xl font-bold">{result.session.user_weight} kg</p>
              </div>
              <div>
                <p className="text-blue-100 text-sm">Age</p>
                <p className="text-2xl font-bold">{result.session.user_age} years</p>
              </div>
              <div>
                <p className="text-blue-100 text-sm">Gender</p>
                <p className="text-2xl font-bold capitalize">{result.session.user_gender}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Tabs defaultValue="measurements" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2 md:grid-cols-4 h-auto">
            <TabsTrigger value="measurements" className="py-3">
              Measurements
            </TabsTrigger>
            <TabsTrigger value="3d-model" className="py-3">
              3D Model
            </TabsTrigger>
            <TabsTrigger value="composition" className="py-3">
              Body Composition
            </TabsTrigger>
            <TabsTrigger value="images" className="py-3">
              Analysis Images
            </TabsTrigger>
          </TabsList>

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
                  {result.measurements.map((measurement) => {
                    const Icon = measurementIcons[measurement.measurement_type] || Ruler;
                    return (
                      <MeasurementCard
                        key={measurement.id}
                        icon={Icon}
                        label={measurement.measurement_type}
                        value={measurement.value}
                        referenceMin={measurement.reference_min}
                        referenceMax={measurement.reference_max}
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
                  <Button variant="outline" size="sm">
                    <Download className="w-4 h-4 mr-2" />
                    Download Mesh
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="bg-gray-100 rounded-lg aspect-video flex items-center justify-center">
                  <div className="text-center">
                    <Activity className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600 font-medium">3D Model Viewer</p>
                    <p className="text-sm text-gray-500 mt-2">
                      Interactive 3D visualization would appear here
                    </p>
                    <p className="text-xs text-gray-400 mt-4">
                      Rotate, zoom, and view from different angles
                    </p>
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-2 mt-4">
                  <Button variant="outline" size="sm">
                    Front View
                  </Button>
                  <Button variant="outline" size="sm">
                    Side View
                  </Button>
                  <Button variant="outline" size="sm">
                    Back View
                  </Button>
                  <Button variant="outline" size="sm">
                    360° Spin
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="composition" className="space-y-6">
            {result.bodyComposition && (
              <BodyCompositionChart bodyComposition={result.bodyComposition} />
            )}
          </TabsContent>

          <TabsContent value="images" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Analysis Images</CardTitle>
              </CardHeader>
              <CardContent>
                {result.images.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {result.images.map((image) => (
                      <div key={image.id}>
                        <p className="text-sm font-medium text-gray-700 mb-2 capitalize">
                          {image.image_type.replace('_', ' ')}
                        </p>
                        <div className="bg-gray-100 rounded-lg aspect-[3/4] flex items-center justify-center">
                          <p className="text-gray-500 text-sm">Image placeholder</p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="bg-gray-50 rounded-lg p-8 text-center">
                    <p className="text-gray-600">
                      No annotated images available for this analysis.
                    </p>
                  </div>
                )}
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
                  Your body analysis is complete! You can download your results, schedule regular
                  check-ins to track progress, or consult with a professional for personalized
                  recommendations.
                </p>
                <div className="flex flex-wrap gap-3">
                  <Link href="/analyze">
                    <Button size="sm">Start New Analysis</Button>
                  </Link>
                  <Button size="sm" variant="outline">
                    Download Report
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
