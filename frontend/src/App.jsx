import { useState, useCallback } from "react";

const API = "http://localhost:8000";

const CLASS_META = {
  0: { name: "Background",        color: "#0a0a14", border: "#3a4a5a" },
  1: { name: "Necrotic Core",     color: "#cc2200", border: "#ff5533" },
  2: { name: "Peritumoral Edema", color: "#00aa44", border: "#00ee66" },
  3: { name: "Enhancing Tumor",   color: "#ccaa00", border: "#ffdd00" },
};

const COLOR_MAP = {
  0: [10,  10,  20],
  1: [255, 50,  20],
  2: [0,   220, 80],
  3: [255, 220, 0],
};

const REGIONS = [
  { key: "WT", label: "Whole Tumor",   desc: "Labels 1+2+3", color: "#4fa3e0" },
  { key: "TC", label: "Tumor Core",    desc: "Labels 1+3",   color: "#ff5533" },
  { key: "ET", label: "Enhancing",     desc: "Label 3",      color: "#ffdd00" },
];

// ── MRI grayscale canvas ──────────────────────────────────────────────────────
function MRICanvas({ data, label }) {
  const ref = useCallback((canvas) => {
    if (!canvas || !data || data.length === 0) return;
    const h = data.length, w = data[0].length;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = data[y][x];
        const i = (y * w + x) * 4;
        img.data[i] = v; img.data[i+1] = v; img.data[i+2] = v; img.data[i+3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data]);

  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:6 }}>
      <span style={{ fontSize:11, letterSpacing:"0.15em", color:"#7ab8d8", textTransform:"uppercase" }}>{label}</span>
      <canvas ref={ref} style={{ width:160, height:160, imageRendering:"pixelated", border:"1px solid #1e3a50", borderRadius:4, background:"#000" }} />
    </div>
  );
}

// ── Segmentation color canvas ─────────────────────────────────────────────────
function SegCanvas({ data, label }) {
  const ref = useCallback((canvas) => {
    if (!canvas || !data || data.length === 0) return;
    const h = data.length, w = data[0].length;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const val = data[y][x];
        const [r, g, b] = COLOR_MAP[val] || [0,0,0];
        const i = (y * w + x) * 4;
        img.data[i] = r; img.data[i+1] = g; img.data[i+2] = b;
        img.data[i+3] = val === 0 ? 40 : 230;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data]);

  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:6 }}>
      <span style={{ fontSize:11, letterSpacing:"0.15em", color:"#7ab8d8", textTransform:"uppercase" }}>{label}</span>
      <canvas ref={ref} style={{ width:160, height:160, imageRendering:"pixelated", border:"1px solid #1e3a50", borderRadius:4, background:"#050810" }} />
    </div>
  );
}

// ── MRI + segmentation overlay canvas ────────────────────────────────────────
function OverlayCanvas({ mriData, segData, label }) {
  const ref = useCallback((canvas) => {
    if (!canvas || !mriData || !segData || mriData.length === 0) return;
    const h = mriData.length, w = mriData[0].length;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const gray = mriData[y][x];
        const seg  = segData[y][x];
        const i    = (y * w + x) * 4;
        if (seg === 0) {
          img.data[i] = gray; img.data[i+1] = gray; img.data[i+2] = gray; img.data[i+3] = 255;
        } else {
          const [r, g, b] = COLOR_MAP[seg];
          img.data[i]   = Math.round(gray * 0.45 + r * 0.55);
          img.data[i+1] = Math.round(gray * 0.45 + g * 0.55);
          img.data[i+2] = Math.round(gray * 0.45 + b * 0.55);
          img.data[i+3] = 255;
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [mriData, segData]);

  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:6 }}>
      <span style={{ fontSize:11, letterSpacing:"0.15em", color:"#00ee66", textTransform:"uppercase" }}>{label}</span>
      <canvas ref={ref} style={{ width:160, height:160, imageRendering:"pixelated", border:"1px solid #2a4a30", borderRadius:4, background:"#000" }} />
    </div>
  );
}

// ── File drop zone ────────────────────────────────────────────────────────────
function DropZone({ label, file, onChange }) {
  const [drag, setDrag] = useState(false);
  return (
    <label
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); onChange(e.dataTransfer.files[0]); }}
      style={{
        display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center",
        border:`1px dashed ${drag ? "#4fa3e0" : file ? "#00ee66" : "#2a3a50"}`,
        borderRadius:6, padding:"14px 10px", cursor:"pointer",
        background: drag ? "rgba(79,163,224,0.07)" : file ? "rgba(0,238,102,0.05)" : "rgba(255,255,255,0.02)",
        transition:"all 0.2s ease", minHeight:72,
      }}
    >
      <input type="file" accept=".nii,.nii.gz" onChange={e => onChange(e.target.files[0])} style={{ display:"none" }} />
      <div style={{ fontSize:12, letterSpacing:"0.12em", color:"#7ab8d8", marginBottom:4, fontWeight:500 }}>{label}</div>
      {file
        ? <div style={{ fontSize:11, color:"#00ee66", textAlign:"center", wordBreak:"break-all" }}>✓ {file.name}</div>
        : <div style={{ fontSize:11, color:"#4a6070" }}>drop .nii file</div>
      }
    </label>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [files, setFiles]     = useState({ flair:null, t1:null, t1ce:null, t2:null });
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [plane, setPlane]     = useState("axial");

  const allLoaded = Object.values(files).every(Boolean);

  const runDemo = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch(`${API}/segment/demo`, { method:"POST" });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch(e) { setError(e.message); }
    setLoading(false);
  };

  const runSegment = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const fd = new FormData();
      fd.append("flair", files.flair);
      fd.append("t1",    files.t1);
      fd.append("t1ce",  files.t1ce);
      fd.append("t2",    files.t2);
      const res = await fetch(`${API}/segment`, { method:"POST", body:fd });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch(e) { setError(e.message); }
    setLoading(false);
  };

  const classes   = result?.classes    || {};
  const regions   = result?.regions    || {};
  const slices    = result?.slices     || {};
  const mriSlices = result?.mri_slices || {};

  return (
    <div style={{ minHeight:"100vh", background:"#030609", color:"#d0e4f0", fontFamily:"'IBM Plex Mono', monospace", display:"flex", flexDirection:"column" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* ── Header ── */}
      <header style={{ borderBottom:"1px solid #0f1e2e", padding:"16px 28px", display:"flex", alignItems:"center", justifyContent:"space-between", background:"rgba(5,8,16,0.8)" }}>
        <div>
          <div style={{ display:"flex", alignItems:"center", gap:10 }}>
            <div style={{ width:8, height:8, borderRadius:"50%", background:"#4fa3e0", boxShadow:"0 0 12px #4fa3e0" }} />
            <span style={{ fontSize:15, letterSpacing:"0.2em", color:"#6bbfe8", fontWeight:500 }}>BraTS</span>
            <span style={{ color:"#2a3a50", fontSize:16 }}>·</span>
            <span style={{ fontSize:14, letterSpacing:"0.15em", color:"#9abcd4" }}>3D U-NET SEGMENTATION</span>
          </div>
          <div style={{ fontSize:11, color:"#4a6070", marginTop:4, letterSpacing:"0.12em" }}>
            BraTS2020 · PyTorch · FastAPI · React
          </div>
        </div>
        <div style={{ fontSize:12, color:"#6a8a9a", fontFamily:"monospace" }}>
          {result && !result.demo && `tumor burden: ${result["tumor_burden_%"]}%`}
          {result?.demo && <span style={{ color:"#cc8833" }}>⚠ synthetic data</span>}
        </div>
      </header>

      <main style={{ display:"flex", flex:1 }}>

        {/* ── Left Panel ── */}
        <div style={{ width:264, flexShrink:0, borderRight:"1px solid #0f1e2e", padding:"20px 16px", display:"flex", flexDirection:"column", gap:12, background:"rgba(4,7,14,0.6)" }}>
          <div style={{ fontSize:11, letterSpacing:"0.2em", color:"#5a7a8a", marginBottom:2 }}>── INPUT MODALITIES</div>

          {["flair","t1","t1ce","t2"].map(mod => (
            <DropZone key={mod} label={mod.toUpperCase()} file={files[mod]}
              onChange={f => setFiles(p => ({ ...p, [mod]:f }))} />
          ))}

          <button onClick={runSegment} disabled={!allLoaded || loading} style={{
            padding:"12px", marginTop:4,
            background: allLoaded && !loading ? "#0d2844" : "#090f1a",
            border:`1px solid ${allLoaded && !loading ? "#4fa3e0" : "#1a2a3a"}`,
            borderRadius:6, color: allLoaded && !loading ? "#7ac8f0" : "#2a3a50",
            letterSpacing:"0.15em", fontSize:12, fontFamily:"'IBM Plex Mono', monospace",
            cursor: allLoaded && !loading ? "pointer" : "not-allowed", transition:"all 0.2s", fontWeight:500,
          }}>
            {loading ? "PROCESSING..." : "RUN SEGMENTATION"}
          </button>

          <div style={{ display:"flex", alignItems:"center", gap:8 }}>
            <div style={{ flex:1, height:1, background:"#0f1e2e" }} />
            <span style={{ fontSize:11, color:"#2a3a50" }}>or</span>
            <div style={{ flex:1, height:1, background:"#0f1e2e" }} />
          </div>

          <button onClick={runDemo} disabled={loading} style={{
            padding:10, background:"transparent", border:"1px solid #1a2e40", borderRadius:6,
            color:"#4a7090", letterSpacing:"0.12em", fontSize:11,
            fontFamily:"'IBM Plex Mono', monospace",
            cursor: loading ? "not-allowed" : "pointer", transition:"all 0.2s",
          }}
            onMouseOver={e => { e.currentTarget.style.color="#7ac8f0"; e.currentTarget.style.borderColor="#2a4a60"; }}
            onMouseOut={e  => { e.currentTarget.style.color="#4a7090"; e.currentTarget.style.borderColor="#1a2e40"; }}
          >
            DEMO (SYNTHETIC)
          </button>

          {error && (
            <div style={{ padding:10, background:"rgba(200,40,20,0.1)", border:"1px solid #4a1a10", borderRadius:6, fontSize:11, color:"#e06650", lineHeight:1.7 }}>
              ⚠ {error}
            </div>
          )}

          {/* Legend */}
          <div style={{ marginTop:10 }}>
            <div style={{ fontSize:11, letterSpacing:"0.2em", color:"#5a7a8a", marginBottom:10 }}>── LEGEND</div>
            {Object.entries(CLASS_META).map(([id, c]) => (
              <div key={id} style={{ display:"flex", alignItems:"center", gap:9, marginBottom:8 }}>
                <div style={{ width:10, height:10, borderRadius:2, background:c.color, border:`1px solid ${c.border}`, flexShrink:0 }} />
                <span style={{ fontSize:12, color:"#8aaabb" }}>{c.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── Center Panel ── */}
        <div style={{ flex:1, padding:"20px 24px", display:"flex", flexDirection:"column", gap:16, minWidth:0 }}>

          {/* Empty state */}
          {!result && !loading && (
            <div style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:16 }}>
              <div style={{ width:80, height:80, borderRadius:"50%", border:"1px solid #1a3050", display:"flex", alignItems:"center", justifyContent:"center", background:"radial-gradient(circle, #0a1828 0%, #030609 100%)", boxShadow:"0 0 30px rgba(79,163,224,0.05)" }}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#2a5070" strokeWidth="1.2">
                  <path d="M12 2a9 9 0 0 1 9 9c0 3.5-2 6.5-5 8.1V20H8v-.9C5 17.5 3 14.5 3 11a9 9 0 0 1 9-9z"/>
                  <circle cx="12" cy="11" r="3"/>
                </svg>
              </div>
              <div style={{ textAlign:"center" }}>
                <div style={{ fontSize:14, color:"#5a8090", letterSpacing:"0.1em" }}>Upload 4 MRI modalities or run demo</div>
                <div style={{ fontSize:12, color:"#3a5060", marginTop:6, letterSpacing:"0.08em" }}>FLAIR · T1 · T1ce · T2</div>
              </div>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:20 }}>
              <div style={{ width:48, height:48, border:"2px solid #0f2030", borderTop:"2px solid #4fa3e0", borderRadius:"50%", animation:"spin 1s linear infinite" }} />
              <div style={{ fontSize:13, color:"#4a7a90", letterSpacing:"0.2em" }}>RUNNING INFERENCE...</div>
            </div>
          )}

          {/* Results */}
          {result && !loading && (
            <>
              <div style={{ fontSize:11, letterSpacing:"0.2em", color:"#5a7a8a" }}>── SEGMENTATION OUTPUT</div>

              {/* Plane selector */}
              <div style={{ display:"flex", gap:1, background:"#0a1520", borderRadius:6, padding:3, width:"fit-content" }}>
                {["axial","coronal","sagittal"].map(p => (
                  <button key={p} onClick={() => setPlane(p)} style={{
                    padding:"6px 16px",
                    background: plane===p ? "#0d2844" : "transparent",
                    border: plane===p ? "1px solid #2a5070" : "1px solid transparent",
                    borderRadius:4, color: plane===p ? "#7ac8f0" : "#4a6a7a",
                    fontSize:11, letterSpacing:"0.12em",
                    fontFamily:"'IBM Plex Mono', monospace", cursor:"pointer", transition:"all 0.15s",
                  }}>
                    {p.toUpperCase()}
                  </button>
                ))}
              </div>

              {/* Three-column comparison */}
              <div style={{ background:"#050810", border:"1px solid #0f1e2e", borderRadius:8, padding:20 }}>
                {/* Column headers */}
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:16, marginBottom:12 }}>
                  {["FLAIR MRI", "SEGMENTATION", "OVERLAY"].map(h => (
                    <div key={h} style={{ fontSize:10, letterSpacing:"0.2em", color:"#5a7a8a", textAlign:"center" }}>{h}</div>
                  ))}
                </div>

                {/* Three canvases */}
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:16, justifyItems:"center" }}>

                  {/* MRI */}
                  {mriSlices[plane]
                    ? <MRICanvas data={mriSlices[plane]} label={plane} />
                    : <div style={{ width:160, height:160, background:"#0a1020", borderRadius:4, border:"1px solid #1e2d40", display:"flex", alignItems:"center", justifyContent:"center" }}>
                        <span style={{ fontSize:11, color:"#2a3a50" }}>no MRI data</span>
                      </div>
                  }

                  {/* Segmentation */}
                  {slices[plane] && <SegCanvas data={slices[plane]} label={plane} />}

                  {/* Overlay */}
                  {slices[plane] && mriSlices[plane]
                    ? <OverlayCanvas mriData={mriSlices[plane]} segData={slices[plane]} label={plane} />
                    : slices[plane] && <SegCanvas data={slices[plane]} label={plane} />
                  }

                </div>
              </div>

              {/* Volume info + region cards */}
              <div style={{ display:"flex", gap:10, alignItems:"stretch" }}>
                <div style={{ padding:"14px 18px", background:"#050810", border:"1px solid #0f1e2e", borderRadius:6, minWidth:160 }}>
                  <div style={{ fontSize:11, color:"#5a7a8a", marginBottom:6 }}>Volume</div>
                  <div style={{ fontSize:13, color:"#6a8a9a", fontFamily:"monospace", marginBottom:12 }}>{result.shape?.join(" × ")}</div>
                  <div style={{ fontSize:11, color:"#5a7a8a", marginBottom:4 }}>Tumor burden</div>
                  <div style={{ fontSize:26, color:"#4fa3e0", fontFamily:"monospace", fontWeight:300 }}>
                    {result["tumor_burden_%"]}<span style={{ fontSize:13, marginLeft:3 }}>%</span>
                  </div>
                  {result.demo && <div style={{ marginTop:10, fontSize:10, color:"#aa7722", border:"1px solid #2a1a08", borderRadius:4, padding:"5px 8px" }}>⚠ SYNTHETIC</div>}
                </div>

                {REGIONS.map(r => (
                  <div key={r.key} style={{ flex:1, padding:"14px 16px", background:"#050810", border:`1px solid ${r.color}33`, borderRadius:6 }}>
                    <div style={{ fontSize:10, letterSpacing:"0.18em", color:r.color, marginBottom:6 }}>{r.key}</div>
                    <div style={{ fontSize:14, color:"#c0d8e8", marginBottom:2, fontWeight:500 }}>{r.label}</div>
                    <div style={{ fontSize:11, color:"#5a7a8a", marginBottom:10 }}>{r.desc}</div>
                    <div style={{ fontSize:22, color:r.color, fontFamily:"monospace", fontWeight:300 }}>
                      {(regions[r.key] ?? 0).toLocaleString()}
                      <span style={{ fontSize:11, color:"#5a7a8a", marginLeft:5 }}>vox</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        {/* ── Right Panel ── */}
        <div style={{ width:224, flexShrink:0, borderLeft:"1px solid #0f1e2e", padding:"20px 16px", background:"rgba(4,7,14,0.6)" }}>
          <div style={{ fontSize:11, letterSpacing:"0.2em", color:"#5a7a8a", marginBottom:16 }}>── CLASS BREAKDOWN</div>

          {result
            ? Object.entries(classes).map(([id, info]) => {
                const meta = CLASS_META[parseInt(id)];
                return (
                  <div key={id} style={{ marginBottom:18 }}>
                    <div style={{ fontSize:12, color:meta?.border, marginBottom:5, fontWeight:500 }}>{info.name}</div>
                    <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                      <span style={{ fontSize:11, color:"#7a9aaa" }}>{info.voxels.toLocaleString()}</span>
                      <span style={{ fontSize:11, color:"#9abccc", fontFamily:"monospace" }}>{info.percentage}%</span>
                    </div>
                    <div style={{ height:3, background:"#0a1520", borderRadius:2, overflow:"hidden" }}>
                      <div style={{ height:"100%", width:`${Math.min(info.percentage * 10, 100)}%`, background:meta?.border, borderRadius:2, transition:"width 1.2s ease" }} />
                    </div>
                  </div>
                );
              })
            : <div style={{ fontSize:12, color:"#3a5060", lineHeight:1.8 }}>Run segmentation<br/>to see results</div>
          }

          <div style={{ marginTop:20, paddingTop:16, borderTop:"1px solid #0f1e2e" }}>
            <div style={{ fontSize:11, letterSpacing:"0.2em", color:"#5a7a8a", marginBottom:12 }}>── MODEL</div>
            {[["Type","3D U-Net"],["Input","4 × 128³"],["Classes","4"],["Params","26.3M"],["Device","CUDA"]].map(([k,v]) => (
              <div key={k} style={{ display:"flex", justifyContent:"space-between", marginBottom:7 }}>
                <span style={{ fontSize:11, color:"#4a6a7a" }}>{k}</span>
                <span style={{ fontSize:11, color:"#7a9aaa", fontFamily:"monospace" }}>{v}</span>
              </div>
            ))}
          </div>
        </div>

      </main>
    </div>
  );
}