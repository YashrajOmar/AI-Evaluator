"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import Editor from '@monaco-editor/react';
import Vapi from "@vapi-ai/web";

const publicKey = process.env.NEXT_PUBLIC_VAPI_PUBLIC_KEY || "";
const assistantId = process.env.NEXT_PUBLIC_VAPI_ASSISTANT_ID || "";
const groqApiKey = process.env.NEXT_PUBLIC_GROQ_API_KEY || ""; 

const BACKEND_HTTP = process.env.NEXT_PUBLIC_BACKEND_HTTP || "http://127.0.0.1:8000";
const BACKEND_WS = process.env.NEXT_PUBLIC_BACKEND_WS || "ws://127.0.0.1:8000";

interface VapiMessage {
  type?: string;
  transcriptType?: string;
  role?: string;
  transcript?: string;
  [key: string]: unknown;
}

interface FinalFeedback {
  overall_score: number;
  content_score: number;
  composure_score: number;
  technical_feedback: string;
  behavioral_feedback: string;
  key_strengths: string[];
  areas_for_improvement: string[];
}

export default function InterviewPage() {
  const [role, setRole] = useState("Software Development Engineer (SDE)");
  const [status, setStatus] = useState("Disconnected");
  const [isConnecting, setIsConnecting] = useState(false);
  const [isActive, setIsActive] = useState(false);
  const [confidenceScore, setConfidenceScore] = useState<number | null>(null);
  
  const [volumeLevel, setVolumeLevel] = useState(0);
  const [timer, setTimer] = useState(0);
  const [codeOutput, setCodeOutput] = useState<string>("Ready to compile...");
  const [isCompiling, setIsCompiling] = useState(false);
  const [splitRatio, setSplitRatio] = useState(40); 
  
  const [isCameraHidden, setIsCameraHidden] = useState(false);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [finalFeedback, setFinalFeedback] = useState<FinalFeedback | null>(null);
  const [isGeneratingFeedback, setIsGeneratingFeedback] = useState(false);

  // --- STANDARD REFS ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const editorRef = useRef<{ getValue: () => string } | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const vapiRef = useRef<Vapi | null>(null);

  // --- STATE MACHINE REFS (The single source of truth for cleanup) ---
  const latestLogRef = useRef<{ role: string; text: string }[]>([]);
  const latestScoresRef = useRef<number[]>([]);
  const latestTimerRef = useRef<number>(0);
  const roleRef = useRef<string>(role); 
  const isFinishingRef = useRef<boolean>(false); 

  // Keep refs perfectly in sync with React state
  useEffect(() => { latestTimerRef.current = timer; }, [timer]);
  useEffect(() => { roleRef.current = role; }, [role]);

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  // ==========================================
  // CORE CLEANUP & EVALUATION FUNCTION
  // ==========================================
  const finishInterview = useCallback(async () => {
    // 1. Mutex Lock to prevent duplicate API calls
    if (isFinishingRef.current) return;
    isFinishingRef.current = true;

    setIsGeneratingFeedback(true);
    setStatus("Wrapping up and generating report...");

    // 2. Hard stop all connections
    vapiRef.current?.stop();
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }
    if (timerRef.current) clearInterval(timerRef.current);
    if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);

    // 3. Pull absolute latest data from refs
    const finalTime = latestTimerRef.current;
    const currentRole = roleRef.current;
    const logToEvaluate = latestLogRef.current;
    const scoresToEvaluate = latestScoresRef.current;

    // Edge case: Cancelled before it really started
    if (logToEvaluate.length === 0 && finalTime < 5) {
      setStatus("Interview cancelled (no data).");
      setIsGeneratingFeedback(false);
      setIsActive(false);
      setIsConnecting(false);
      isFinishingRef.current = false; // Unlock so they can restart
      return;
    }

    const avgComposure = scoresToEvaluate.length > 0 
      ? scoresToEvaluate.reduce((a, b) => a + b, 0) / scoresToEvaluate.length 
      : 0.75;
    const composurePercent = Math.round(avgComposure * 100);

    const systemPrompt = `You are an expert technical interviewer and recruiter. 
Evaluate the candidate for the role of ${currentRole} based on BOTH the spoken transcript AND their real-time visual analysis.

CRITICAL VISUAL METRIC:
The candidate's AI facial/posture analysis yielded an average visual composure score of ${composurePercent}%.
- You MUST factor this ${composurePercent}% score heavily into the final 'overall_score'.
- You MUST address their body language and visual confidence directly in the 'behavioral_feedback'.
- If the score is below 60%, note that they appeared visually nervous, distracted, or lacked eye contact, and add this to 'areas_for_improvement'.
- If the score is above 80%, praise their strong visual presence and confident posture in 'key_strengths'.

The interview lasted ${formatTime(finalTime)}.

You MUST return a JSON object with EXACTLY these keys and appropriate data types:
{
  "overall_score": <number 0-100, weighted mix of technical answers and the ${composurePercent}% visual score>,
  "content_score": <number 0-100, based STRICTLY on transcript answers>,
  "composure_score": ${composurePercent},
  "technical_feedback": "<detailed string evaluating technical answers>",
  "behavioral_feedback": "<detailed string evaluating soft skills AND their ${composurePercent}% visual composure score>",
  "key_strengths": ["<strength1>", "<strength2>"],
  "areas_for_improvement": ["<improvement1>", "<improvement2>"]
}`;

    try {
      const response = await fetch(`${BACKEND_HTTP}/evaluate-session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          role: currentRole,
          duration_seconds: finalTime,
          average_composure: parseFloat(avgComposure.toFixed(2)),
          transcript: logToEvaluate,
          system_prompt_instruction: systemPrompt 
        }),
      });
      
      const feedbackData = await response.json();
      setFinalFeedback(feedbackData);
      setShowFeedbackModal(true);
      setStatus("Evaluation Complete.");
      
      localStorage.removeItem('cachedTranscript');
      
    } catch (error) {
      console.error("Failed to generate interview feedback:", error);
      setStatus("Failed to generate feedback. Local data saved.");
    } finally {
      setIsGeneratingFeedback(false);
      setIsActive(false);
      setIsConnecting(false);
      setVolumeLevel(0);
    }
  }, []);

  // ==========================================
  // GRACEFUL TIMEOUT MANAGER
  // ==========================================
  // ==========================================
  // GRACEFUL TIMEOUT MANAGER (60 MINUTE INTERVIEW)
  // ==========================================
  useEffect(() => {
    if (!isActive) return;

    if (timer === 3300) { // 55 minutes
      try {
        vapiRef.current?.send({
          type: "add-message",
          message: {
            role: "system",
            content: "We have about 5 minutes remaining in the interview. Wrap up the current topic, ask the candidate if they have any final questions, and prepare to end the interview."
          }
        });
      } catch (err) {
        console.warn("Failed to send time warning to Vapi:", err);
      }
    } else if (timer >= 3600) { // 60 minutes
      finishInterview();
    }
  }, [timer, isActive, finishInterview]);

  // ==========================================
  // SYSTEM INITIALIZATION & WEBRTC
  // ==========================================
  useEffect(() => {
    if (!vapiRef.current) {
        vapiRef.current = new Vapi(publicKey);
    }

    const activeVideoElement = videoRef.current;

    const initializeMedia = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        if (videoRef.current) videoRef.current.srcObject = stream;

        wsRef.current = new WebSocket(`${BACKEND_WS}/ws/inference`);
        
        wsRef.current.onmessage = (event) => {
          const data = JSON.parse(event.data);
          setConfidenceScore(data.confidence);
          latestScoresRef.current.push(data.confidence);
        };

        const sendFrame = () => {
          if (wsRef.current?.readyState === WebSocket.OPEN && videoRef.current) {
            const video = videoRef.current;
            if (video.videoWidth === 0 || video.videoHeight === 0) return;

            const canvas = document.createElement("canvas");
            const targetWidth = 320;
            const scale = targetWidth / video.videoWidth;
            canvas.width = targetWidth;
            canvas.height = video.videoHeight * scale;
            
            const ctx = canvas.getContext("2d");
            if (ctx) {
              ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
              const base64Frame = canvas.toDataURL("image/jpeg", 0.6);
              wsRef.current.send(base64Frame);
            }
          }
        };

        frameIntervalRef.current = setInterval(sendFrame, 3000);

      } catch (error) {
        console.error("Camera error:", error);
      }
    };

    initializeMedia();

    const onCallStart = () => {
      setStatus("Interview in progress...");
      setIsConnecting(false);
      setIsActive(true);
      timerRef.current = setInterval(() => setTimer((prev) => prev + 1), 1000);
    };
    
    // Unified exits
    const onCallEnd = () => {
      console.log("Vapi trigger: Call ended normally");
      finishInterview();
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const onError = (e: any) => {
      const errType = e?.error?.type || e?.type || "";
      const errMsg = e?.error?.msg || e?.msg || e?.message || "";
      
      // Only terminate on fatal ejection errors. Ignore transient WebSocket/Device warnings.
      if (errType === "ejected" || errType === "call-ended" || errMsg.includes("Meeting has ended")) {
        console.error("Fatal Vapi Ejection triggered:", errType, errMsg);
        finishInterview();
      } else {
        console.warn("Non-fatal Vapi issue ignored:", e);
      }
    };

    const onMessage = (message: VapiMessage) => {
      if (message.type === 'transcript' && message.transcriptType === 'final') {
        const text = message.transcript || '';
        console.log(`[Transcript] ${message.role}: ${text}`);
        
        latestLogRef.current = [
          ...latestLogRef.current,
          { role: message.role === 'assistant' ? 'Interviewer' : 'Candidate', text }
        ];
        
        // Failsafe cache
        localStorage.setItem('cachedTranscript', JSON.stringify(latestLogRef.current));

        // Intercept AI's completion signal
        if (message.role === 'assistant' && text.includes('<INTERVIEW_COMPLETE>')) {
          console.log("AI emitted interview completion signal.");
          finishInterview();
        }
      }
    };

    vapiRef.current.on("call-start", onCallStart);
    vapiRef.current.on("call-end", onCallEnd);
    vapiRef.current.on("error", onError);
    vapiRef.current.on("volume-level", setVolumeLevel);
    vapiRef.current.on("message", onMessage);

    return () => {
      vapiRef.current?.removeAllListeners();
      wsRef.current?.close();
      if (timerRef.current) clearInterval(timerRef.current);
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
      
      if (activeVideoElement?.srcObject) {
        (activeVideoElement.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
    };
  }, [finishInterview]);

  const handleStart = async () => {
    // 1. Check if keys are loaded
    if (!publicKey || !assistantId) {
      setStatus("Error: Keys missing");
      alert("Missing keys! Please stop your terminal (Ctrl+C) and run 'npm run dev' again to load the .env.local file.");
      return;
    }

    isFinishingRef.current = false;
    setIsConnecting(true);
    setStatus("Connecting to AI...");
    setTimer(0);
    latestLogRef.current = [];
    latestScoresRef.current = [];
    latestTimerRef.current = 0;

    try {
      // 2. Connect to your Vapi dashboard assistant and pass the selected role
      await vapiRef.current?.start(assistantId, {
        variableValues: { role: role }
      });
    } catch (error) {
      console.error("Failed to start Vapi call:", error);
      setStatus("Connection failed.");
      setIsConnecting(false);
    }
  };



  
  const handleEditorDidMount = (editor: { getValue: () => string }) => {
    editorRef.current = editor;
  };

  const runCode = async () => {
    if (!editorRef.current) return;
    const code = editorRef.current.getValue();
    setIsCompiling(true);
    setCodeOutput("Compiling and running...");

    try {
      const response = await fetch(`${BACKEND_HTTP}/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      const data = await response.json();
      setCodeOutput(data.output);
    } catch (error) {
      console.error("Execution failure:", error);
      setCodeOutput("Failed to connect to backend server. Is FastAPI running?");
    } finally {
      setIsCompiling(false);
    }
  };

  const togglePictureInPicture = async () => {
    if (!videoRef.current) return;
    if (document.pictureInPictureElement) {
      await document.exitPictureInPicture();
    } else {
      await videoRef.current.requestPictureInPicture();
    }
  };

  const toggleFullscreen = async () => {
    if (!videoRef.current) return;
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await videoRef.current.requestFullscreen();
    }
  };

  const resetModal = () => {
    setShowFeedbackModal(false);
    setStatus("Disconnected");
    isFinishingRef.current = false; 
  };

  const isCodingRole = [
    "Software Development Engineer (SDE)", 
    "Backend Engineer", 
    "Frontend Engineer", 
    "Full Stack", 
    "Data Science", 
    "Machine Learning / AI", 
    "Data Analytics",
    "Embedded Systems",
    "Cybersecurity"
  ].includes(role);
  
  const isDesignRole = ["System Design", "Cloud Engineer", "DevOps", "Product Management (PM)", "UI/UX Designer"].includes(role);
  const showSplitter = isCodingRole || isDesignRole;

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col p-4 font-sans relative">
      <div className="bg-gray-800 p-4 rounded-xl shadow-lg w-full flex flex-col gap-4 mb-4 border border-gray-700">
        <div className="flex flex-wrap justify-between items-center">
          <div className="flex items-center gap-4">
              <h2 className="text-xl font-bold">AI Interview Evaluator</h2>
              <div className="text-sm px-3 py-1 bg-blue-900 text-blue-200 rounded font-medium">
                {status}
              </div>
              <div className="text-xl font-mono text-gray-300 font-bold border border-gray-600 px-3 py-1 rounded bg-black">
                {formatTime(timer)}
              </div>
          </div>
          
          <div className="flex items-center gap-4">
            <select 
              value={role}
              onChange={(e) => setRole(e.target.value)}
              disabled={isActive || isConnecting}
              className="p-2 bg-gray-700 border border-gray-600 rounded-md focus:outline-none disabled:opacity-50 text-sm"
            >
              <option>Software Development Engineer (SDE)</option>
              <option>Backend Engineer</option>
              <option>Frontend Engineer</option>
              <option>Full Stack</option>
              <option>System Design</option>
              <option>Cloud Engineer</option>
              <option>DevOps</option>
              <option>QA Automation</option>
              <option>Embedded Systems</option>
              <option>Cybersecurity</option>
              <option>Mobile Developer</option>
              <option>Product Management (PM)</option>
              <option>UI/UX Designer</option>
              <option>Data Science</option>
              <option>Machine Learning / AI</option>
              <option>Data Analytics</option>
              <option>HR (Behavioral Only)</option>
            </select>
            
            <div className="flex gap-2">
              <button 
                onClick={handleStart} 
                disabled={isActive || isConnecting || isGeneratingFeedback} 
                className="px-4 py-2 bg-green-600 text-white font-bold rounded-md hover:bg-green-700 disabled:opacity-50"
              >
                Start
              </button>
              <button 
                onClick={() => finishInterview()} 
                disabled={!isActive && !isConnecting} 
                className="px-4 py-2 bg-red-600 text-white font-bold rounded-md hover:bg-red-700 disabled:opacity-50"
              >
                {isGeneratingFeedback ? "Wrapping up..." : "Stop"}
              </button>
            </div>
          </div>
        </div>

        {showSplitter && (
          <div className="flex items-center justify-center gap-4 w-full border-t border-gray-700 pt-3">
            <span className="text-xs font-bold text-gray-400 uppercase">AI Panel</span>
            <input 
              type="range" 
              min="20" 
              max="80" 
              value={splitRatio} 
              onChange={(e) => setSplitRatio(Number(e.target.value))}
              className="w-64 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
            />
            <span className="text-xs font-bold text-gray-400 uppercase">Workspace</span>
          </div>
        )}
      </div>

      <div className="flex flex-1 gap-4 h-[calc(100vh-180px)] relative">
        <div 
          style={{ width: showSplitter ? `${splitRatio}%` : '100%' }} 
          className="transition-all duration-100 ease-linear flex flex-col gap-4"
        >
          <div className="bg-black flex-1 rounded-xl shadow-lg flex flex-col items-center justify-center overflow-hidden border border-gray-700 relative">
             <div className="flex flex-col items-center justify-center relative z-10">
                <div className={`w-24 h-24 rounded-full bg-gray-800 flex items-center justify-center border-4 ${isActive ? 'border-blue-500 animate-pulse' : 'border-gray-600'}`}>
                  <span className="text-4xl">🤖</span>
                </div>
                
                <div className="flex gap-1 items-end h-10 mt-6">
                  {[1, 2, 3, 4, 5, 6, 7].map((bar) => (
                    <div 
                      key={bar} 
                      className="w-2 bg-blue-500 rounded-t transition-all duration-100 ease-in"
                      style={{ height: `${isActive ? Math.max(10, volumeLevel * 100 * (1 - (Math.abs(4 - bar) * 0.15))) : 10}%` }}
                    />
                  ))}
                </div>
                <p className="mt-4 text-gray-400 font-medium">{isActive ? 'AI is listening...' : 'Ready to start'}</p>
             </div>

             <div className="absolute top-4 left-4 bg-black/60 backdrop-blur px-3 py-1 rounded border border-gray-600 z-20">
                <span className="text-xs font-bold text-gray-300">Confidence: </span>
                <span className={`font-mono font-bold text-sm ${confidenceScore && confidenceScore > 0.5 ? 'text-green-400' : 'text-red-400'}`}>
                  {confidenceScore ? confidenceScore.toFixed(2) : "Analyzing..."}
                </span>
             </div>
          </div>
        </div>

        {isCodingRole && (
          <div style={{ width: `${100 - splitRatio}%` }} className="transition-all duration-100 ease-linear flex flex-col gap-4">
            <div className="bg-gray-800 flex-1 rounded-xl shadow-lg border border-gray-700 overflow-hidden pt-4">
              <Editor 
                height="100%" 
                defaultLanguage="cpp" 
                theme="vs-dark" 
                defaultValue={`#include <iostream>\nusing namespace std;\n\nint main() {\n    cout << "Ready to solve!";\n    return 0;\n}`}
                onMount={handleEditorDidMount}
              />
            </div>
            
            <div className="bg-gray-800 h-1/3 rounded-xl shadow-lg border border-gray-700 p-4 flex flex-col">
              <div className="flex justify-between items-center mb-2">
                <span className="font-bold text-gray-300 uppercase text-xs tracking-wider">Terminal</span>
                <button onClick={runCode} disabled={isCompiling} className="bg-blue-600 hover:bg-blue-700 px-4 py-1 rounded text-sm font-bold disabled:opacity-50">
                  {isCompiling ? "Running..." : "Run Code"}
                </button>
              </div>
              <pre className="bg-black flex-1 rounded p-3 text-sm text-green-400 font-mono overflow-y-auto whitespace-pre-wrap">
                {codeOutput}
              </pre>
            </div>
          </div>
        )}

        {!isCameraHidden ? (
          <div className="absolute bottom-4 left-4 z-50 shadow-2xl rounded-lg overflow-hidden border-2 border-gray-600 transition-all duration-200 group">
             <video 
               ref={videoRef} 
               autoPlay 
               muted 
               playsInline 
               className="w-48 h-36 object-cover transform scale-x-[-1] bg-black" 
             />
             
             <div className="absolute top-0 left-0 w-full h-full bg-black/50 backdrop-blur-sm flex items-center justify-center gap-3 transition-opacity duration-200 opacity-0 group-hover:opacity-100">
                <button onClick={togglePictureInPicture} title="Pop Out" className="p-2 bg-gray-800 hover:bg-blue-600 rounded-full text-white">📍</button>
                <button onClick={toggleFullscreen} title="Fullscreen" className="p-2 bg-gray-800 hover:bg-blue-600 rounded-full text-white">⛶</button>
                <button onClick={() => setIsCameraHidden(true)} title="Hide Camera" className="p-2 bg-gray-800 hover:bg-red-600 rounded-full text-white">✖</button>
             </div>
          </div>
        ) :  (
           <button 
             onClick={() => setIsCameraHidden(false)}
             className="absolute bottom-4 left-4 z-50 p-3 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-full shadow-lg transition-all"
             title="Show Camera"
           >
             📷 Show
           </button>
        )}
        
        {isCameraHidden && (
           <video ref={videoRef} autoPlay muted playsInline className="opacity-0 pointer-events-none absolute w-0 h-0" />
        )}
      </div>

      {showFeedbackModal && finalFeedback && (
        <div className="absolute inset-0 bg-black/80 backdrop-blur-sm z-[100] flex items-center justify-center p-4 lg:p-8 overflow-y-auto">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl w-full max-w-4xl p-6 lg:p-10 flex flex-col gap-6">
            
            <div className="flex justify-between items-start border-b border-gray-700 pb-4">
              <div>
                <h2 className="text-3xl font-bold text-white mb-1">Interview Evaluation Report</h2>
                <p className="text-gray-400">Role: {role} | Duration: {formatTime(timer)}</p>
              </div>
              <button 
                onClick={resetModal}
                className="text-gray-400 hover:text-white text-2xl font-bold"
              >
                ×
              </button>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 flex flex-col items-center justify-center">
                <span className="text-gray-400 font-bold uppercase text-xs mb-1">Overall Fit</span>
                <span className="text-4xl font-bold text-blue-400">{finalFeedback.overall_score}/100</span>
              </div>
              <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 flex flex-col items-center justify-center">
                <span className="text-gray-400 font-bold uppercase text-xs mb-1">Content Score</span>
                <span className="text-4xl font-bold text-green-400">{finalFeedback.content_score}/100</span>
              </div>
              <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 flex flex-col items-center justify-center">
                <span className="text-gray-400 font-bold uppercase text-xs mb-1">Composure Score</span>
                <span className="text-4xl font-bold text-purple-400">{finalFeedback.composure_score}%</span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-2">
              <div className="flex flex-col gap-2">
                <h3 className="text-lg font-bold text-gray-200 border-b border-gray-700 pb-2">Technical Analysis</h3>
                <p className="text-gray-400 text-sm leading-relaxed whitespace-pre-wrap">{finalFeedback.technical_feedback}</p>
              </div>
              <div className="flex flex-col gap-2">
                <h3 className="text-lg font-bold text-gray-200 border-b border-gray-700 pb-2">Behavioral Analysis</h3>
                <p className="text-gray-400 text-sm leading-relaxed whitespace-pre-wrap">{finalFeedback.behavioral_feedback}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-green-900/20 border border-green-800/50 p-4 rounded-xl">
                <h3 className="text-green-400 font-bold mb-3">Key Strengths</h3>
                <ul className="list-disc pl-5 flex flex-col gap-1">
                  {finalFeedback.key_strengths.map((item: string, idx: number) => (
                    <li key={idx} className="text-gray-300 text-sm">{item}</li>
                  ))}
                </ul>
              </div>
              <div className="bg-red-900/20 border border-red-800/50 p-4 rounded-xl">
                <h3 className="text-red-400 font-bold mb-3">Areas for Improvement</h3>
                <ul className="list-disc pl-5 flex flex-col gap-1">
                  {finalFeedback.areas_for_improvement.map((item: string, idx: number) => (
                    <li key={idx} className="text-gray-300 text-sm">{item}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="flex justify-end mt-4">
              <button 
                onClick={resetModal}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition-colors"
              >
                Close Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}