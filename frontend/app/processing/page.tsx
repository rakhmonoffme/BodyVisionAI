'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { ProgressBar } from '@/components/visualization/progress-bar';
import { useAppStore } from '@/store/app-store';
import { Loader2, X } from 'lucide-react';

export default function ProcessingPage() {
  const router = useRouter();
  const { processing, resetAnalysis } = useAppStore();

  useEffect(() => {
    if (!processing.sessionId && processing.status === 'idle') {
      router.push('/analyze');
      return;
    }

    if (processing.status === 'completed' && processing.sessionId) {
      router.push(`/results/${processing.sessionId}`);
      return;
    }

    if (processing.status === 'error') {
      // Stay on page to show error
      return;
    }
  }, [processing.status, processing.sessionId, router]);

  const handleCancel = () => {
    resetAnalysis();
    router.push('/');
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white flex items-center justify-center p-4">
      <Card className="max-w-3xl w-full">
        <CardContent className="p-8">
          <div className="text-center mb-8">
            {processing.status === 'error' ? (
              <>
                <div className="flex items-center justify-center mb-4">
                  <X className="w-16 h-16 text-red-600" />
                </div>
                <h1 className="text-3xl font-bold text-gray-900 mb-2">Analysis Failed</h1>
                <p className="text-gray-600">{processing.currentStep || 'An error occurred'}</p>
              </>
            ) : (
              <>
                <div className="flex items-center justify-center mb-4">
                  <Loader2 className="w-16 h-16 text-blue-600 animate-spin" />
                </div>
                <h1 className="text-3xl font-bold text-gray-900 mb-2">Processing Your Analysis</h1>
                <p className="text-gray-600">
                  Please wait 30-60 seconds while we analyze your photos
                </p>
              </>
            )}
          </div>

          {processing.status !== 'error' && (
            <ProgressBar
              progress={processing.progress}
              currentStep={processing.currentStep}
              className="mb-8"
            />
          )}

          <div className="flex justify-center gap-4">
            {processing.status === 'error' ? (
              <>
                <Button variant="outline" onClick={handleCancel}>
                  Back to Home
                </Button>
                <Button onClick={() => router.push('/analyze')}>
                  Try Again
                </Button>
              </>
            ) : (
              <Button variant="outline" onClick={handleCancel}>
                <X className="mr-2 w-4 h-4" />
                Cancel Analysis
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}