'use client';
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, CheckCircle2, Activity, Layers, ActivitySquare, Loader2, Search, Compass, Image as ImageIcon, FileText } from 'lucide-react';
import { useStore } from '../store/useStore';
import { getPrediction } from '../services/getPrediction';

const MODELS = [
  { id: 'keypoint_rcnn', name: 'Keypoint R-CNN', description: 'Best for precise joint detection' },
  { id: 'yolov8_pose', name: 'YOLOv8 Pose', description: 'Fast and robust tracking' },
  { id: 'vertenet', name: 'VerteNet', description: 'Specialized for spine tracking' },
  { id: 'residual_unet', name: 'Residual-Unet', description: 'High precision segmentation' }
];

export default function Sidebar() {
  const { 
    imageFile, selectedModel, prediction, isLoading, 
    setImage, setSelectedModel, setPrediction, setIsLoading,
    displayDetections, showDetectionLabels, detectionScale,
    displayKeypoints, keypointMode, keypointColor, keypointRadius,
    displayCobbAngle,
    setDisplayDetections, setShowDetectionLabels, setDetectionScale,
    setDisplayKeypoints, setKeypointMode, setKeypointColor, setKeypointRadius,
    setDisplayCobbAngle
  } = useStore();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const url = URL.createObjectURL(file);
      setImage(file, url);
    }
  }, [setImage]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    maxFiles: 1,
  });

  const handlePredict = async () => {
    if (!imageFile) return;
    setIsLoading(true);
    try {
      const result = await getPrediction(imageFile, selectedModel);
      setPrediction(result);
    } catch (error) {
      console.error('Prediction failed', error);
      alert('Failed to get prediction. Ensure backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderAngleCard = (label: string, angleInfo: any, colorClass: string, isPrimary = false) => {
    if (!angleInfo) return null;
    const { angle, idxs } = angleInfo;
    
    let severity = 'Normal';
    if (angle > 10 && angle <= 25) severity = 'Mild';
    else if (angle > 25 && angle <= 40) severity = 'Moderate';
    else if (angle > 40) severity = 'Severe';
    
    return (
      <div className={`p-4 rounded-xl border ${isPrimary ? 'bg-slate-800/80 border-cyan-500/50' : 'bg-slate-800/40 border-slate-700/50'} backdrop-blur-sm transition-all hover:bg-slate-800`}>
        <div className="flex justify-between items-start mb-2">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${colorClass}`}></div>
            <span className="font-semibold text-slate-200">{label} Curve</span>
          </div>
          <span className={`text-xs px-2 py-1 rounded-md font-medium ${
            severity === 'Normal' ? 'bg-green-500/20 text-green-400' :
            severity === 'Mild' ? 'bg-yellow-500/20 text-yellow-400' :
            severity === 'Moderate' ? 'bg-orange-500/20 text-orange-400' :
            'bg-red-500/20 text-red-400'
          }`}>
            {severity}
          </span>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-3xl font-bold text-white">{angle.toFixed(1)}&deg;</span>
          {angle > 0 && <span className="text-sm text-slate-400">T{idxs[0] + 1} - T{idxs[1] + 1}</span>}
        </div>
      </div>
    );
  };

  return (
    <div className="w-full md:w-96 h-full bg-slate-950 border-r border-slate-800/50 p-6 flex flex-col gap-6 overflow-y-auto custom-scrollbar">
      <div>
        <h1 className="text-3xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500 mb-1">
          ScolioVis
        </h1>
        <p className="text-sm text-slate-400">Advanced deep learning Cobb Angle analysis.</p>
      </div>

      {/* Upload Zone */}
      <div
        {...getRootProps()}
        className={`relative group p-6 rounded-2xl border-2 border-dashed transition-all cursor-pointer overflow-hidden ${
          isDragActive ? 'border-cyan-400 bg-cyan-400/10' : 'border-slate-700 hover:border-cyan-500/50 hover:bg-slate-900/50'
        }`}
      >
        <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
        <input {...getInputProps()} />
        <div className="flex flex-col items-center justify-center text-center gap-3 relative z-10">
          <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center group-hover:scale-110 transition-transform">
            <UploadCloud className="text-cyan-400" size={24} />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-200">
              {isDragActive ? 'Drop X-Ray here...' : 'Upload X-Ray Image'}
            </p>
            <p className="text-xs text-slate-500 mt-1">Drag & drop or click to browse</p>
          </div>
        </div>
      </div>

      {/* Model Selection */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-slate-300 font-medium">
          <Layers size={18} className="text-indigo-400" />
          <h2>AI Model</h2>
        </div>
        <div className="grid grid-cols-1 gap-2">
          {MODELS.map((model) => (
            <button
              key={model.id}
              onClick={() => setSelectedModel(model.id)}
              className={`text-left p-3 rounded-xl border transition-all ${
                selectedModel === model.id
                  ? 'bg-indigo-500/10 border-indigo-500/50'
                  : 'bg-slate-900/40 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`font-medium ${selectedModel === model.id ? 'text-indigo-300' : 'text-slate-300'}`}>
                  {model.name}
                </span>
                {selectedModel === model.id && <CheckCircle2 size={16} className="text-indigo-400" />}
              </div>
              <p className="text-xs text-slate-500 mt-1">{model.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Visual Controls Section */}
      {imageFile && (
        <div className="space-y-4 pt-4 border-t border-slate-800/50">
          <div className="flex items-center gap-2 text-slate-300 font-medium mb-1">
            <ImageIcon size={18} className="text-cyan-400" />
            <div className="flex flex-col">
              <span className="text-xs text-slate-500">Input Image</span>
              <span className="text-sm font-semibold text-slate-200">{imageFile.name}</span>
            </div>
          </div>

          <div className="space-y-4 bg-slate-900/40 p-4 rounded-xl border border-slate-800">
            {/* Display Detections Switch */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-3 cursor-pointer select-none text-sm font-medium text-slate-300">
                  <button
                    type="button"
                    onClick={() => setDisplayDetections(!displayDetections)}
                    className={`w-11 h-6 flex items-center rounded-full p-1 transition-colors ${displayDetections ? 'bg-blue-600' : 'bg-slate-700'}`}
                  >
                    <div className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${displayDetections ? 'translate-x-5' : 'translate-x-0'}`} />
                  </button>
                  <Search size={16} className="text-slate-400" />
                  Display Detections
                </label>
              </div>

              {displayDetections && (
                <div className="pl-14 space-y-3 text-xs text-slate-400">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showDetectionLabels}
                      onChange={(e) => setShowDetectionLabels(e.target.checked)}
                      className="rounded border-slate-700 bg-slate-800 text-blue-600 focus:ring-0 w-4 h-4"
                    />
                    Show Detection Labels
                  </label>
                  <div className="flex items-center gap-3">
                    <span>Scale</span>
                    <input
                      type="range"
                      min="1"
                      max="5"
                      value={detectionScale}
                      onChange={(e) => setDetectionScale(Number(e.target.value))}
                      className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="border-t border-slate-800/60 my-2" />

            {/* Display Keypoints Switch */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-3 cursor-pointer select-none text-sm font-medium text-slate-300">
                  <button
                    type="button"
                    onClick={() => setDisplayKeypoints(!displayKeypoints)}
                    className={`w-11 h-6 flex items-center rounded-full p-1 transition-colors ${displayKeypoints ? 'bg-blue-600' : 'bg-slate-700'}`}
                  >
                    <div className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${displayKeypoints ? 'translate-x-5' : 'translate-x-0'}`} />
                  </button>
                  <Compass size={16} className="text-slate-400" />
                  Display Keypoints
                </label>
              </div>

              {displayKeypoints && (
                <div className="pl-14 space-y-3 text-xs text-slate-400">
                  {/* Mode Selector */}
                  <div className="flex items-center justify-between gap-2">
                    <span>Mode</span>
                    <div className="flex items-center bg-slate-800/80 p-0.5 rounded-lg border border-slate-700">
                      <button
                        type="button"
                        onClick={() => setKeypointMode('dots')}
                        title="Dots Only"
                        className={`p-1.5 rounded-md transition-all ${keypointMode === 'dots' ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
                      >
                        <svg className="w-4 h-4 fill-current" viewBox="0 0 16 16">
                          <circle cx="4" cy="4" r="1.5" /><circle cx="12" cy="4" r="1.5" />
                          <circle cx="4" cy="12" r="1.5" /><circle cx="12" cy="12" r="1.5" />
                        </svg>
                      </button>
                      <button
                        type="button"
                        onClick={() => setKeypointMode('horizontal')}
                        title="Horizontal Lines"
                        className={`p-1.5 rounded-md transition-all ${keypointMode === 'horizontal' ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
                      >
                        <svg className="w-4 h-4 stroke-current stroke-2" viewBox="0 0 16 16" fill="none">
                          <line x1="3" y1="4" x2="13" y2="4" /><line x1="3" y1="12" x2="13" y2="12" />
                          <circle cx="3" cy="4" r="1" fill="currentColor" /><circle cx="13" cy="4" r="1" fill="currentColor" />
                          <circle cx="3" cy="12" r="1" fill="currentColor" /><circle cx="13" cy="12" r="1" fill="currentColor" />
                        </svg>
                      </button>
                      <button
                        type="button"
                        onClick={() => setKeypointMode('connected')}
                        title="Connected Lines"
                        className={`p-1.5 rounded-md transition-all ${keypointMode === 'connected' ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
                      >
                        <svg className="w-4 h-4 stroke-current stroke-2" viewBox="0 0 16 16" fill="none">
                          <line x1="4" y1="3" x2="4" y2="13" /><line x1="12" y1="3" x2="12" y2="13" />
                          <circle cx="4" cy="3" r="1" fill="currentColor" /><circle cx="12" cy="3" r="1" fill="currentColor" />
                          <circle cx="4" cy="13" r="1" fill="currentColor" /><circle cx="12" cy="13" r="1" fill="currentColor" />
                        </svg>
                      </button>
                      <button
                        type="button"
                        onClick={() => setKeypointMode('quadrilateral')}
                        title="Quadrilateral"
                        className={`p-1.5 rounded-md transition-all ${keypointMode === 'quadrilateral' ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
                      >
                        <svg className="w-4 h-4 stroke-current stroke-2" viewBox="0 0 16 16" fill="none">
                          <rect x="3" y="3" width="10" height="10" rx="1" />
                        </svg>
                      </button>
                    </div>
                  </div>

                  {/* Colors */}
                  <div className="flex items-center gap-3">
                    <span>Colors</span>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setKeypointColor('#ffffff')}
                        className={`w-6 h-6 rounded-md bg-white border-2 transition-all ${keypointColor === '#ffffff' ? 'border-blue-500 scale-110' : 'border-transparent'}`}
                      />
                      <button
                        type="button"
                        onClick={() => setKeypointColor('#00ffff')}
                        className={`w-6 h-6 rounded-md bg-cyan-400 border-2 transition-all ${keypointColor === '#00ffff' ? 'border-blue-500 scale-110' : 'border-transparent'}`}
                      />
                    </div>
                  </div>

                  {/* Radius */}
                  <div className="flex items-center gap-3">
                    <span>Radius</span>
                    <input
                      type="range"
                      min="1"
                      max="6"
                      value={keypointRadius}
                      onChange={(e) => setKeypointRadius(Number(e.target.value))}
                      className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="border-t border-slate-800/60 my-2" />

            {/* Display Cobb Angle Switch */}
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-3 cursor-pointer select-none text-sm font-medium text-slate-300">
                <button
                  type="button"
                  onClick={() => setDisplayCobbAngle(!displayCobbAngle)}
                  className={`w-11 h-6 flex items-center rounded-full p-1 transition-colors ${displayCobbAngle ? 'bg-blue-600' : 'bg-slate-700'}`}
                >
                  <div className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${displayCobbAngle ? 'translate-x-5' : 'translate-x-0'}`} />
                </button>
                <FileText size={16} className="text-slate-400" />
                Display Cobb Angle
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Action Button */}
      <button
        onClick={handlePredict}
        disabled={!imageFile || isLoading}
        className={`w-full py-3.5 rounded-xl font-medium transition-all flex items-center justify-center gap-2 ${
          !imageFile 
            ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
            : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-lg shadow-cyan-500/25 active:scale-[0.98]'
        }`}
      >
        {isLoading ? (
          <>
            <Loader2 className="animate-spin" size={18} />
            Processing...
          </>
        ) : (
          <>
            <ActivitySquare size={18} />
            Analyze Spine
          </>
        )}
      </button>

      {/* Results Section */}
      {prediction && (
        <div className="space-y-4 pt-4 border-t border-slate-800/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-slate-300 font-medium">
              <Activity size={18} className="text-emerald-400" />
              <h2>Analysis Results</h2>
            </div>
            <div className="px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs font-semibold text-slate-300">
              {prediction.curve_type}-Curve Detected
            </div>
          </div>
          
          <div className="flex flex-col gap-3">
            {renderAngleCard('PT (Proximal)', prediction.angles_with_pos.pt, 'bg-orange-500')}
            {renderAngleCard('MT (Main)', prediction.angles_with_pos.mt, 'bg-pink-500', true)}
            {renderAngleCard('LT / TL (Lumbar)', prediction.angles_with_pos.lt, 'bg-green-500')}
          </div>
        </div>
      )}
    </div>
  );
}
