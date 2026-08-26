import React, { useState, useRef } from 'react';

export default function UploadModal({ onClose, onUploadComplete }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [gsd, setGsd] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleFile = f => {
    if (!f) return;
    if (!f.type.startsWith('image/')) { setError('Please select a JPG/PNG image'); return; }
    if (f.size > 25 * 1024 * 1024) { setError('File too large (>25MB)'); return; }
    setError('');
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
  };

  const onDrop = e => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    handleFile(f);
  };

  const runInference = async () => {
    if (!file) { setError('Select an image first'); return; }
    setLoading(true); setError('');
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('gsd', String(gsd));
      // try backend first
      let res;
      try {
        res = await fetch('/api/infer', { method: 'POST', body: fd });
      } catch (e) {
        // fallback direct to localhost:8000 if proxy not configured
        res = await fetch('http://localhost:8000/api/infer', { method: 'POST', body: fd });
      }
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Server error ${res.status}`);
      }
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      // build data URLs
      const inputDataUrl = `data:image/jpeg;base64,${data.input_b64}`;
      const overlayDataUrl = `data:image/jpeg;base64,${data.overlay_b64}`;
      onUploadComplete({
        imageDataUrl: inputDataUrl,
        overlayDataUrl,
        width: data.width,
        height: data.height,
        gsd: data.gsd,
        buildings: data.buildings || [],
        roads: data.roads || [],
        water: data.water || [],
        counts: data.counts,
        elapsed: data.elapsed_sec,
      });
      onClose();
    } catch (e) {
      setError(e.message || String(e));
      // If backend not running, give helpful hint
      if (e.message.includes('Failed to fetch') || e.message.includes('NetworkError')) {
        setError('Backend not running. Start it:  python3 ml/src/serve.py  (or uvicorn ml.src.serve:app --port 8000)');
      }
    } finally { setLoading(false); }
  };

  return (
    <div className="fixed inset-0 z-[3000] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4" onClick={onClose}>
      <div className="gcs-panel w-full max-w-[640px] max-h-[90vh] overflow-y-auto flex flex-col" onClick={e=>e.stopPropagation()}>
        <div className="flex items-center justify-between px-4 py-3 border-b border-gcs-border">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-gcs-cyan animate-pulse" />
            <span className="font-mono text-xs font-bold tracking-widest text-gcs-cyan">UPLOAD CUSTOM IMAGE</span>
            <span className="font-mono text-[10px] text-gcs-dim ml-2">JPG/PNG · auto-detects roads/buildings/water</span>
          </div>
          <button onClick={onClose} className="font-mono text-gcs-dim hover:text-white text-lg leading-none px-2">×</button>
        </div>

        <div className="p-4 space-y-4">
          {/* Drop zone */}
          <div
            onDragOver={e=>{e.preventDefault(); setDragOver(true);}}
            onDragLeave={()=>setDragOver(false)}
            onDrop={onDrop}
            onClick={()=>inputRef.current?.click()}
            className={`border-2 border-dashed rounded flex flex-col items-center justify-center p-6 cursor-pointer transition-colors ${dragOver ? 'border-gcs-cyan bg-gcs-cyan/10' : 'border-gcs-border hover:border-gcs-cyan/50 bg-slate-900/30'}`}
          >
            {preview ? (
              <img src={preview} alt="preview" className="max-h-[260px] max-w-full object-contain rounded border border-gcs-border" />
            ) : (
              <>
                <span className="font-mono text-sm text-gcs-cyan">Drop image here or click to browse</span>
                <span className="font-mono text-xs text-gcs-dim mt-1">Orthophoto / drone image, 512×512 to 3000×3000 px</span>
              </>
            )}
            <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/jpg" className="hidden" onChange={e=>handleFile(e.target.files?.[0])} />
            {file && <span className="font-mono text-xs text-gcs-dim mt-2">{file.name} · {(file.size/1024/1024).toFixed(2)} MB</span>}
          </div>

          {/* GSD selector */}
          <div className="flex items-center gap-3">
            <label className="font-mono text-xs text-gcs-dim">GSD (m/px)</label>
            <select value={gsd} onChange={e=>setGsd(parseFloat(e.target.value))} className="bg-slate-900 border border-gcs-border text-gcs-text font-mono text-xs px-2 py-1 rounded">
              <option value={0.5}>0.5 — SVAMITVA drone (real)</option>
              <option value={1.0}>1.0 — high-res aerial</option>
              <option value={1.5}>1.5 — demo tile (default)</option>
              <option value={2.0}>2.0 — satellite</option>
            </select>
            <span className="font-mono text-xs text-gcs-dim">affects area/length calc</span>
          </div>

          {error && <div className="font-mono text-xs text-[#ff3355] bg-[#ff3355]/10 border border-[#ff3355]/30 px-3 py-2 rounded">{error}</div>}

          <div className="flex gap-2">
            <button
              onClick={runInference}
              disabled={!file || loading}
              className={`flex-1 font-mono text-xs font-bold tracking-widest py-2.5 border transition-colors ${!file||loading ? 'border-gcs-border text-gcs-dim bg-slate-900/30 cursor-not-allowed' : 'border-gcs-cyan text-gcs-cyan hover:bg-gcs-cyan hover:text-black'}`}
            >
              {loading ? 'RUNNING INFERENCE…' : '▶ RUN DETECTION'}
            </button>
            <button onClick={onClose} className="px-4 py-2.5 font-mono text-xs text-gcs-dim border border-gcs-border hover:text-white">CANCEL</button>
          </div>

          <div className="font-mono text-[11px] text-gcs-dim leading-4 border-t border-gcs-border pt-3">
            <div className="font-bold text-gcs-text mb-1">How it works:</div>
            • Roads use improved V3 detector (<code>ml/src/road_detector.py:24</code>) — vegetation mask + width filter, no GPU needed.<br/>
            • Buildings use Attention-ResUNet (<code>ml/src/model.py:1</code>) if checkpoint found, else roads+water only.<br/>
            • CLI alternative: <code>python3 ml/src/infer_upload.py your.jpg --gsd 0.5 --out outputs/upload</code><br/>
            • Result opens as new map layer — toggle BUILDINGS/ROADS/WATER as usual.
          </div>
        </div>
      </div>
    </div>
  );
}
