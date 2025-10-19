'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Camera, Ruler, Activity, Shield, ArrowRight } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <nav className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <Activity className="w-8 h-8 text-blue-600" />
              <span className="text-xl font-bold text-gray-900">BodyAnalyzer</span>
            </div>
            <Link href="/analyze">
              <Button size="lg">Start Analysis</Button>
            </Link>
          </div>
        </div>
      </nav>

      <main>
        <section className="py-20 px-4 sm:px-6 lg:px-8">
          <div className="max-w-7xl mx-auto text-center">
            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
              AI-Powered Body Analysis
              <br />
              <span className="text-blue-600">in 3 Simple Steps</span>
            </h1>
            <p className="text-xl text-gray-600 mb-10 max-w-3xl mx-auto">
              Get accurate body measurements, 3D visualization, and comprehensive body composition
              analysis using advanced AI technology.
            </p>
            <Link href="/analyze">
              <Button size="lg" className="text-lg px-8 py-6 h-auto">
                Start Your Analysis
                <ArrowRight className="ml-2 w-5 h-5" />
              </Button>
            </Link>
          </div>
        </section>

        <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
          <div className="max-w-7xl mx-auto">
            <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
              Powerful Features
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <Card className="border-2 hover:border-blue-500 transition-colors">
                <CardContent className="p-6 text-center">
                  <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Ruler className="w-8 h-8 text-blue-600" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">
                    Accurate Measurements
                  </h3>
                  <p className="text-gray-600">
                    Get precise measurements of 10+ body parts including neck, shoulders, chest,
                    waist, hips, and more.
                  </p>
                </CardContent>
              </Card>

              <Card className="border-2 hover:border-blue-500 transition-colors">
                <CardContent className="p-6 text-center">
                  <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Camera className="w-8 h-8 text-green-600" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">3D Body Model</h3>
                  <p className="text-gray-600">
                    Visualize your body in 3D with interactive controls. Rotate, zoom, and view
                    from any angle.
                  </p>
                </CardContent>
              </Card>

              <Card className="border-2 hover:border-blue-500 transition-colors">
                <CardContent className="p-6 text-center">
                  <div className="w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Activity className="w-8 h-8 text-orange-600" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">
                    Body Composition Analysis
                  </h3>
                  <p className="text-gray-600">
                    Understand your body fat percentage, lean mass, BMI, and body type
                    classification.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        <section className="py-16 px-4 sm:px-6 lg:px-8">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">How It Works</h2>
            <div className="space-y-8">
              {[
                {
                  step: '1',
                  title: 'Upload Photos',
                  description:
                    'Take or upload front, side, and back photos of yourself in form-fitting clothes against a plain background.',
                },
                {
                  step: '2',
                  title: 'Enter Information',
                  description:
                    'Provide your height, weight, age, and gender for accurate analysis and personalized results.',
                },
                {
                  step: '3',
                  title: 'Get Results',
                  description:
                    'Receive detailed measurements, interactive 3D model, and comprehensive body composition analysis in minutes.',
                },
              ].map((item) => (
                <div key={item.step} className="flex gap-6 items-start">
                  <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center text-xl font-bold flex-shrink-0">
                    {item.step}
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">{item.title}</h3>
                    <p className="text-gray-600">{item.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-16 px-4 sm:px-6 lg:px-8 bg-blue-600 text-white">
          <div className="max-w-5xl mx-auto">
            <div className="flex items-start gap-6">
              <Shield className="w-12 h-12 flex-shrink-0" />
              <div>
                <h2 className="text-2xl font-bold mb-4">Privacy & Security</h2>
                <p className="text-blue-100 mb-4">
                  Your privacy is our top priority. All photos and data are processed securely and
                  are never shared with third parties. You have full control over your data and can
                  delete it at any time.
                </p>
                <ul className="space-y-2 text-blue-100">
                  <li>• End-to-end encrypted data transmission</li>
                  <li>• No data sharing with third parties</li>
                  <li>• Complete data deletion on request</li>
                  <li>• HIPAA-compliant infrastructure</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gray-50">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-3xl font-bold text-gray-900 mb-6">Ready to Get Started?</h2>
            <p className="text-xl text-gray-600 mb-8">
              Begin your body analysis journey today and unlock detailed insights about your
              physical health.
            </p>
            <Link href="/analyze">
              <Button size="lg" className="text-lg px-8 py-6 h-auto">
                Start Your Analysis Now
                <ArrowRight className="ml-2 w-5 h-5" />
              </Button>
            </Link>
          </div>
        </section>
      </main>

      <footer className="border-t bg-white py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center text-gray-600">
          <p>&copy; 2025 BodyAnalyzer. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
