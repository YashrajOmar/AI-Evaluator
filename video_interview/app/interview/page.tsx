'use client';

import { useEffect, useRef, useState } from 'react';

export default function InterviewRoom() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [hasPermission, setHasPermission] = useState<boolean>(false);

  useEffect(() => {
    async function setupCamera() {
      try {
        // Request access to both video and audio
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: true, 
          audio: true 
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setHasPermission(true);
        }
      } catch (error) {
        console.error("Error accessing media devices:", error);
        alert("Please allow camera and microphone access to start the interview.");
      }
    }
    
    setupCamera();
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-950 text-white p-6">
      
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold mb-2">AI Software Engineer Interview</h1>
        <p className="text-gray-400">Behavioral and Technical Evaluation</p>
      </div>

      {/* Video Container */}
      <div className="relative w-full max-w-3xl aspect-video bg-gray-900 rounded-xl overflow-hidden shadow-2xl border border-gray-800">
        {!hasPermission && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-500 z-10">
            Requesting camera access...
          </div>
        )}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted // Muted to prevent audio feedback loop
          className={`w-full h-full object-cover transition-opacity duration-500 ${hasPermission ? 'opacity-100' : 'opacity-0'}`}
        />
        
        {/* Recording Indicator */}
        {hasPermission && (
          <div className="absolute top-4 right-4 flex items-center gap-2 bg-black/50 px-3 py-1.5 rounded-full backdrop-blur-sm">
            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span className="text-xs font-medium text-white">Live</span>
          </div>
        )}
      </div>

      {/* Placeholder for future AI chat/feedback controls */}
      <div className="mt-8 flex gap-4">
        <button className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors">
          End Interview
        </button>
      </div>
      
    </div>
  );
}