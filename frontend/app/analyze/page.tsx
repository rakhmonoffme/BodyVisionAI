'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Checkbox } from '@/components/ui/checkbox';
import { PhotoUploadZone } from '@/components/upload/photo-upload-zone';
import { FormSlider } from '@/components/forms/form-slider';
import { useAppStore } from '@/store/app-store';
import { PhotoType, Gender } from '@/types';
import { ArrowLeft, ArrowRight, CheckCircle2, Upload, User, FileCheck } from 'lucide-react';
import Link from 'next/link';

export default function AnalyzePage() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [termsAccepted, setTermsAccepted] = useState(false);

  const { photos, userInfo, setPhoto, setUserInfo, submitAnalysis } = useAppStore();

  const handlePhotoSelect = (type: PhotoType, file: File) => {
    setPhoto(type, file);
  };

  const handlePhotoRemove = (type: PhotoType) => {
    setPhoto(type, null as any);
  };

  const canProceedStep1 = photos.front && photos.side && photos.back;
  const canProceedStep2 = userInfo.height && userInfo.weight && userInfo.age && userInfo.gender;
  const canSubmit = canProceedStep1 && canProceedStep2 && termsAccepted;

  const handleSubmit = async () => {
    if (!canSubmit) return;

    // Navigate to processing page IMMEDIATELY
    router.push('/processing');
    
    // Start analysis in background (don't await)
    submitAnalysis().catch((error) => {
      console.error('Analysis failed:', error);
      // Processing page will handle the error state
    });
  };

  const renderStep = () => {
    switch (step) {
      case 1:
        return (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Your Photos</h2>
              <p className="text-gray-600">
                Take or upload clear photos from three angles for accurate analysis
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <PhotoUploadZone
                type="front"
                file={photos.front}
                onFileSelect={(file) => handlePhotoSelect('front', file)}
                onFileRemove={() => handlePhotoRemove('front')}
              />
              <PhotoUploadZone
                type="side"
                file={photos.side}
                onFileSelect={(file) => handlePhotoSelect('side', file)}
                onFileRemove={() => handlePhotoRemove('side')}
              />
              <PhotoUploadZone
                type="back"
                file={photos.back}
                onFileSelect={(file) => handlePhotoSelect('back', file)}
                onFileRemove={() => handlePhotoRemove('back')}
              />
            </div>

            <Card className="bg-blue-50 border-blue-200">
              <CardContent className="p-4">
                <h3 className="font-semibold text-gray-900 mb-3">Photo Guidelines:</h3>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span>Wear form-fitting clothes or minimal clothing</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span>Stand against a plain, solid-colored background</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span>Ensure your full body is visible in each photo</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span>Good lighting with no shadows on your body</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span>Keep arms slightly away from your body</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <div className="flex justify-end">
              <Button size="lg" disabled={!canProceedStep1} onClick={() => setStep(2)}>
                Continue
                <ArrowRight className="ml-2 w-4 h-4" />
              </Button>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Personal Information</h2>
              <p className="text-gray-600">
                Help us provide accurate analysis tailored to you
              </p>
            </div>

            <Card>
              <CardContent className="p-6 space-y-6">
                <FormSlider
                  label="Height"
                  value={userInfo.height}
                  onChange={(value) => setUserInfo({ height: value })}
                  min={100}
                  max={250}
                  unit="cm"
                />

                <FormSlider
                  label="Weight"
                  value={userInfo.weight}
                  onChange={(value) => setUserInfo({ weight: value })}
                  min={30}
                  max={300}
                  unit="kg"
                />

                <FormSlider
                  label="Age"
                  value={userInfo.age}
                  onChange={(value) => setUserInfo({ age: value })}
                  min={10}
                  max={120}
                  unit="years"
                />

                <div className="space-y-3">
                  <Label className="text-sm font-medium">Gender</Label>
                  <RadioGroup
                    value={userInfo.gender}
                    onValueChange={(value: Gender) => setUserInfo({ gender: value })}
                  >
                    <div className="grid grid-cols-3 gap-4">
                      <div className="flex items-center space-x-2 border rounded-lg p-3 hover:bg-gray-50 cursor-pointer">
                        <RadioGroupItem value="male" id="male" />
                        <Label htmlFor="male" className="cursor-pointer flex-1">
                          Male
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2 border rounded-lg p-3 hover:bg-gray-50 cursor-pointer">
                        <RadioGroupItem value="female" id="female" />
                        <Label htmlFor="female" className="cursor-pointer flex-1">
                          Female
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2 border rounded-lg p-3 hover:bg-gray-50 cursor-pointer">
                        <RadioGroupItem value="neutral" id="neutral" />
                        <Label htmlFor="neutral" className="cursor-pointer flex-1">
                          Neutral
                        </Label>
                      </div>
                    </div>
                  </RadioGroup>
                </div>
              </CardContent>
            </Card>

            <div className="flex justify-between">
              <Button variant="outline" size="lg" onClick={() => setStep(1)}>
                <ArrowLeft className="mr-2 w-4 h-4" />
                Back
              </Button>
              <Button size="lg" disabled={!canProceedStep2} onClick={() => setStep(3)}>
                Continue
                <ArrowRight className="ml-2 w-4 h-4" />
              </Button>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Review & Submit</h2>
              <p className="text-gray-600">Check your information before starting the analysis</p>
            </div>

            <Card>
              <CardContent className="p-6 space-y-6">
                <div>
                  <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <Upload className="w-5 h-5" />
                    Uploaded Photos
                  </h3>
                  <div className="grid grid-cols-3 gap-4">
                    {(['front', 'side', 'back'] as PhotoType[]).map((type) => (
                      <div key={type} className="space-y-2">
                        <p className="text-sm font-medium text-gray-700 capitalize">{type} View</p>
                        {photos[type] && (
                          <div className="relative w-full h-32 rounded-lg overflow-hidden border">
                            <img
                              src={URL.createObjectURL(photos[type]!)}
                              alt={`${type} preview`}
                              className="w-full h-full object-cover"
                            />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="border-t pt-6">
                  <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <User className="w-5 h-5" />
                    Personal Information
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <p className="text-sm text-gray-600">Height</p>
                      <p className="text-lg font-semibold text-gray-900">
                        {userInfo.height} cm
                      </p>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <p className="text-sm text-gray-600">Weight</p>
                      <p className="text-lg font-semibold text-gray-900">{userInfo.weight} kg</p>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <p className="text-sm text-gray-600">Age</p>
                      <p className="text-lg font-semibold text-gray-900">
                        {userInfo.age} years
                      </p>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <p className="text-sm text-gray-600">Gender</p>
                      <p className="text-lg font-semibold text-gray-900 capitalize">
                        {userInfo.gender}
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-blue-200 bg-blue-50">
              <CardContent className="p-4">
                <div className="flex items-start gap-3">
                  <Checkbox
                    id="terms"
                    checked={termsAccepted}
                    onCheckedChange={(checked) => setTermsAccepted(checked as boolean)}
                  />
                  <Label htmlFor="terms" className="text-sm text-gray-700 cursor-pointer">
                    I confirm that all information provided is accurate and I consent to the
                    processing of my photos and data for body analysis purposes. I understand that
                    this data will be stored securely and can be deleted upon request.
                  </Label>
                </div>
              </CardContent>
            </Card>

            <div className="flex justify-between">
              <Button variant="outline" size="lg" onClick={() => setStep(2)}>
                <ArrowLeft className="mr-2 w-4 h-4" />
                Back
              </Button>
              <Button size="lg" disabled={!canSubmit} onClick={handleSubmit}>
                <FileCheck className="mr-2 w-4 h-4" />
                Submit Analysis
              </Button>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <nav className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link href="/" className="flex items-center gap-2">
              <ArrowLeft className="w-5 h-5" />
              <span className="font-semibold">Back to Home</span>
            </Link>
          </div>
        </div>
      </nav>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-8">
          <div className="flex items-center justify-center gap-2 mb-8">
            {[1, 2, 3].map((s) => (
              <div key={s} className="flex items-center">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold transition-colors ${
                    s === step
                      ? 'bg-blue-600 text-white'
                      : s < step
                      ? 'bg-green-500 text-white'
                      : 'bg-gray-200 text-gray-600'
                  }`}
                >
                  {s < step ? <CheckCircle2 className="w-5 h-5" /> : s}
                </div>
                {s < 3 && (
                  <div
                    className={`w-16 h-1 mx-2 transition-colors ${
                      s < step ? 'bg-green-500' : 'bg-gray-200'
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600">
              Step {step} of 3:{' '}
              {step === 1 ? 'Photo Upload' : step === 2 ? 'Personal Information' : 'Review'}
            </p>
          </div>
        </div>

        {renderStep()}
      </main>
    </div>
  );
}