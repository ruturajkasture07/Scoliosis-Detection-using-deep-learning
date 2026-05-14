'use client';
import React, { useEffect, useRef } from 'react';
import { useStore } from '../store/useStore';

export default function ImageCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const {
    imageUrl,
    prediction,
    displayDetections,
    showDetectionLabels,
    detectionScale,
    displayKeypoints,
    keypointMode,
    keypointColor,
    keypointRadius,
    displayCobbAngle
  } = useStore();

  useEffect(() => {
    if (!imageUrl || !canvasRef.current || !containerRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      // Scale canvas to fit container while maintaining aspect ratio
      const containerWidth = containerRef.current!.clientWidth;
      const containerHeight = containerRef.current!.clientHeight;
      
      const scale = Math.min(containerWidth / img.width, containerHeight / img.height);
      
      const drawWidth = img.width * scale;
      const drawHeight = img.height * scale;
      
      canvas.width = drawWidth;
      canvas.height = drawHeight;
      
      // Draw image
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, drawWidth, drawHeight);

      if (prediction) {
        const { detections, landmarks, midpoint_lines } = prediction;

        // Ensure canvas line dash is reset
        ctx.setLineDash([]);

        // 1. Draw Bounding Boxes
        if (displayDetections) {
          ctx.strokeStyle = 'rgba(37, 99, 235, 0.8)'; // Tailwind blue-600
          ctx.lineWidth = detectionScale;
          detections.forEach((box, idx) => {
            const [x1, y1, x2, y2] = box;
            ctx.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
            
            if (showDetectionLabels) {
              ctx.fillStyle = 'rgba(37, 99, 235, 0.95)';
              ctx.font = 'bold 11px sans-serif';
              ctx.fillText(`V${idx + 1}`, x1 * scale, (y1 * scale) - 5);
            }
          });
        }

        // 2. Draw Keypoints & Shapes
        if (displayKeypoints) {
          ctx.strokeStyle = keypointColor;
          ctx.lineWidth = 1.5;
          
          for (let i = 0; i < landmarks.length; i += 4) {
            const tl = landmarks[i];
            const tr = landmarks[i + 1];
            const bl = landmarks[i + 2];
            const br = landmarks[i + 3];
            
            if (tl && tr && bl && br) {
              if (keypointMode === 'horizontal') {
                ctx.beginPath();
                ctx.moveTo(tl[0] * scale, tl[1] * scale);
                ctx.lineTo(tr[0] * scale, tr[1] * scale);
                ctx.moveTo(bl[0] * scale, bl[1] * scale);
                ctx.lineTo(br[0] * scale, br[1] * scale);
                ctx.stroke();
              } else if (keypointMode === 'connected') {
                ctx.beginPath();
                ctx.moveTo(tl[0] * scale, tl[1] * scale);
                ctx.lineTo(bl[0] * scale, bl[1] * scale);
                ctx.moveTo(tr[0] * scale, tr[1] * scale);
                ctx.lineTo(br[0] * scale, br[1] * scale);
                ctx.stroke();
              } else if (keypointMode === 'quadrilateral') {
                ctx.beginPath();
                ctx.moveTo(tl[0] * scale, tl[1] * scale);
                ctx.lineTo(tr[0] * scale, tr[1] * scale);
                ctx.lineTo(br[0] * scale, br[1] * scale);
                ctx.lineTo(bl[0] * scale, bl[1] * scale);
                ctx.closePath();
                ctx.stroke();
              }
            }
          }

          // Draw the dot points
          ctx.fillStyle = keypointColor;
          landmarks.forEach((pt) => {
            const [x, y] = pt;
            ctx.beginPath();
            ctx.arc(x * scale, y * scale, keypointRadius, 0, 2 * Math.PI);
            ctx.fill();
          });
        }

        // 3. Draw Cobb Angle lines & Extrapolations continuously
        if (displayCobbAngle) {
          // Draw Mid-line/Tilt Estimation
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)'; // solid white horizontal line
          ctx.lineWidth = 1.5;
          midpoint_lines.forEach((line) => {
            const [p1, p2] = line;
            ctx.beginPath();
            ctx.moveTo(p1[0] * scale, p1[1] * scale);
            ctx.lineTo(p2[0] * scale, p2[1] * scale);
            ctx.stroke();
          });
          
          // Extrapolate lines for regional Cobb angles continuously
          const drawCobbLines = (info: any, color: string) => {
            if (!info || info.idxs[0] === info.idxs[1]) return;
            const v1 = info.idxs[0];
            const v2 = info.idxs[1];
            
            if (midpoint_lines[v1] && midpoint_lines[v2]) {
               const l1 = midpoint_lines[v1];
               const l2 = midpoint_lines[v2];
               
               ctx.strokeStyle = color;
               ctx.lineWidth = 2.5;
               // Continuous line as requested, explicitly overriding any dash
               ctx.setLineDash([]);
               
               const extendLine = (line: number[][], extendBy: number) => {
                  const cx = (line[0][0] + line[1][0]) / 2;
                  const cy = (line[0][1] + line[1][1]) / 2;
                  
                  const dx = line[1][0] - line[0][0];
                  const dy = line[1][1] - line[0][1];
                  
                  const length = Math.sqrt(dx*dx + dy*dy) || 1;
                  const nx = dx / length;
                  const ny = dy / length;
                  
                  ctx.beginPath();
                  ctx.moveTo((cx - nx * extendBy) * scale, (cy - ny * extendBy) * scale);
                  ctx.lineTo((cx + nx * extendBy) * scale, (cy + ny * extendBy) * scale);
                  ctx.stroke();
               };
               
               extendLine(l1, img.width * 0.45);
               extendLine(l2, img.width * 0.45);
            }
          };

          drawCobbLines(prediction.angles_with_pos.pt, '#f97316'); // orange
          drawCobbLines(prediction.angles_with_pos.mt, '#ec4899'); // pink
          drawCobbLines(prediction.angles_with_pos.lt, '#22c55e'); // green
        }
      }
    };
  }, [
    imageUrl,
    prediction,
    displayDetections,
    showDetectionLabels,
    detectionScale,
    displayKeypoints,
    keypointMode,
    keypointColor,
    keypointRadius,
    displayCobbAngle
  ]);

  return (
    <div ref={containerRef} className="w-full h-[70vh] md:h-full bg-slate-900 rounded-xl overflow-hidden shadow-2xl border border-slate-800 flex items-center justify-center relative">
      {!imageUrl ? (
        <div className="text-slate-500 flex flex-col items-center">
          <span className="text-sm">No image selected</span>
        </div>
      ) : (
        <canvas ref={canvasRef} className="max-w-full max-h-full" />
      )}
    </div>
  );
}
