'use client';

import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, CheckCircle2, X } from 'lucide-react';
import { PhotoType } from '@/types';
import { validatePhotoFile } from '@/lib/validation';
import { cn } from '@/lib/utils';

interface PhotoUploadZoneProps {
  type: PhotoType;
  file: File | null;
  onFileSelect: (file: File) => void;
  onFileRemove: () => void;
}

const typeLabels: Record<PhotoType, string> = {
  front: 'Front View',
  side: 'Side View',
  back: 'Back View',
};

export function PhotoUploadZone({ type, file, onFileSelect, onFileRemove }: PhotoUploadZoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const selectedFile = acceptedFiles[0];
        const validation = validatePhotoFile(selectedFile);

        if (validation.valid) {
          onFileSelect(selectedFile);
        } else {
          alert(validation.error);
        }
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/webp': ['.webp'],
    },
    maxFiles: 1,
    multiple: false,
  });

  return (
    <div className="relative">
      <div
        {...getRootProps()}
        className={cn(
          'relative border-2 border-dashed rounded-lg p-6 transition-all cursor-pointer',
          'hover:border-blue-500 hover:bg-blue-50/50',
          isDragActive && 'border-blue-500 bg-blue-50',
          file && 'border-green-500 bg-green-50/30'
        )}
      >
        <input {...getInputProps()} />

        {file ? (
          <div className="space-y-3">
            <div className="flex items-center justify-center">
              <CheckCircle2 className="w-12 h-12 text-green-500" />
            </div>
            <div className="text-center">
              <p className="font-medium text-gray-900">{typeLabels[type]}</p>
              <p className="text-sm text-gray-600 mt-1">{file.name}</p>
              <p className="text-xs text-gray-500 mt-1">
                {(file.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
            <div className="relative w-full h-40 rounded-md overflow-hidden bg-gray-100">
              <img
                src={URL.createObjectURL(file)}
                alt={`${type} preview`}
                className="w-full h-full object-cover"
              />
            </div>
          </div>
        ) : (
          <div className="space-y-3 text-center">
            <div className="flex items-center justify-center">
              <Upload className="w-12 h-12 text-gray-400" />
            </div>
            <div>
              <p className="font-medium text-gray-900">{typeLabels[type]}</p>
              <p className="text-sm text-gray-600 mt-1">
                {isDragActive ? 'Drop image here' : 'Drag & drop or click to upload'}
              </p>
              <p className="text-xs text-gray-500 mt-2">JPEG, PNG, or WebP (max 10MB)</p>
            </div>
          </div>
        )}
      </div>

      {file && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onFileRemove();
          }}
          className="absolute -top-2 -right-2 w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center hover:bg-red-600 transition-colors shadow-lg"
          aria-label="Remove photo"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
}
