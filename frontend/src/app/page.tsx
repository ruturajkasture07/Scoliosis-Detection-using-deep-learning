import Sidebar from '@/components/Sidebar';
import ImageCanvas from '@/components/ImageCanvas';

export default function Home() {
  return (
    <main className="flex h-screen w-full bg-slate-950 text-slate-200 overflow-hidden font-sans">
      <Sidebar />
      <div className="flex-1 p-6 flex flex-col relative">
        <header className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white">Spinal Analysis Viewer</h2>
            <p className="text-sm text-slate-400">Interactive X-Ray analysis & Cobb Angle detection</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-sm bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
              Live Detection
            </div>
          </div>
        </header>
        
        <div className="flex-1 w-full relative">
          <ImageCanvas />
        </div>
      </div>
    </main>
  );
}
