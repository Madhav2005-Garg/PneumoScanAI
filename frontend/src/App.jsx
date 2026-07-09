import { useState, useRef, useCallback, useEffect } from "react";
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadialBarChart, RadialBar, ReferenceLine
} from "recharts";

const API_URL = "http://localhost:5000";

// ── Theme tokens ──────────────────────────────────────────────────────────────
const DARK = {
  bg:         "#080E1A",
  bgGrad:     "linear-gradient(135deg,#080E1A 0%,#0D1829 100%)",
  surface:    "#0F1C2E",
  surfaceHi:  "#162236",
  surfaceHov: "#1C2D44",
  border:     "#1E3050",
  borderHov:  "#2A4570",
  accent:     "#3B82F6",
  accentHov:  "#60A5FA",
  accentDim:  "#1D4ED8",
  accentGlow: "#3B82F630",
  danger:     "#F87171",
  dangerDeep: "#EF4444",
  safe:       "#34D399",
  safeDeep:   "#10B981",
  warn:       "#FBBF24",
  warnDeep:   "#F59E0B",
  purple:     "#C084FC",
  purpleDeep: "#A855F7",
  textPri:    "#F0F6FF",
  textSec:    "#7B9CC4",
  textDim:    "#3D5878",
  shadow:     "0 8px 32px #00000060",
  shadowSm:   "0 2px 8px #00000040",
  glass:      "rgba(15,28,46,0.85)",
};

const LIGHT = {
  bg:         "#F0F4FA",
  bgGrad:     "linear-gradient(135deg,#EEF2FA 0%,#F5F8FF 100%)",
  surface:    "#FFFFFF",
  surfaceHi:  "#F3F7FF",
  surfaceHov: "#E8EFFF",
  border:     "#D4DEEF",
  borderHov:  "#A8BEDF",
  accent:     "#2563EB",
  accentHov:  "#1D4ED8",
  accentDim:  "#3B82F6",
  accentGlow: "#2563EB18",
  danger:     "#DC2626",
  dangerDeep: "#B91C1C",
  safe:       "#059669",
  safeDeep:   "#047857",
  warn:       "#D97706",
  warnDeep:   "#B45309",
  purple:     "#7C3AED",
  purpleDeep: "#6D28D9",
  textPri:    "#0F172A",
  textSec:    "#475569",
  textDim:    "#94A3B8",
  shadow:     "0 8px 32px #00000018",
  shadowSm:   "0 2px 8px #0000000E",
  glass:      "rgba(255,255,255,0.90)",
};

// Training history — ConvNeXt-Tiny, 2-phase fine-tuning, real epoch logs
// Phase 1 (epochs 1-5): backbone frozen, head-only warm-up
// Phase 2 (epochs 6-7): features.6 + features.7 + classifier unfrozen
const PHASE1_EPOCHS = 5;
const TRAIN_HISTORY = {
  loss:     [0.1292, 0.0794, 0.0679, 0.0642, 0.0563, 0.0523, 0.0504],
  val_loss: [0.0755, 0.1103, 0.0842, 0.0719, 0.0845, 0.0643, 0.0675],
  acc:      [0.901,  0.942,  0.946,  0.952,  0.959,  0.960,  0.963],
  val_acc:  [0.925,  0.886,  0.912,  0.934,  0.918,  0.951,  0.945],
  auc:      [0.962,  0.985,  0.989,  0.990,  0.992,  0.994,  0.994],
  val_auc:  [0.987,  0.990,  0.991,  0.992,  0.992,  0.992,  0.993],
};
// Confusion matrix at threshold=0.5 (best balanced performance), n=624 test images
// At 0.5  → TP=371 FP=45  FN=19  TN=189  Acc=90%
// At 0.092 → TP=387 FP=87  FN=3   TN=147  Acc=86% (val-tuned max-F1 threshold)
const CM = { TP:371, FP:45, FN:19, TN:189 };
const DECISION_THRESHOLD = 0.092;  // val-tuned F1-optimal threshold from notebook

// ── Global styles injected once ───────────────────────────────────────────────
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'Inter',sans-serif;}
  ::-webkit-scrollbar{width:6px;height:6px;}
  ::-webkit-scrollbar-track{background:transparent;}
  ::-webkit-scrollbar-thumb{background:#3B82F640;border-radius:3px;}
  @keyframes spin{to{transform:rotate(360deg)}}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
  @keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
  @keyframes slideIn{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:translateX(0)}}
  @keyframes glow{0%,100%{box-shadow:0 0 8px #3B82F640}50%{box-shadow:0 0 24px #3B82F680}}
  @keyframes scanline{0%{top:0%}100%{top:100%}}
  @keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
  .fade-in{animation:fadeIn 0.35s ease both;}
  .slide-in{animation:slideIn 0.3s ease both;}
`;

// ── Shared primitives ─────────────────────────────────────────────────────────
function useTheme() {
  const [dark, setDark] = useState(true);
  return { C: dark ? DARK : LIGHT, dark, toggle: () => setDark(d => !d) };
}

function Badge({ color, children, size="sm" }) {
  const pad = size==="lg" ? "5px 14px" : "3px 10px";
  const fs  = size==="lg" ? 13 : 11;
  return (
    <span style={{
      display:"inline-flex", alignItems:"center", gap:6,
      background:color+"22", border:`1px solid ${color}55`,
      color, borderRadius:20, padding:pad, fontSize:fs, fontWeight:700,
      letterSpacing:"0.04em", textTransform:"uppercase", whiteSpace:"nowrap"
    }}>
      <span style={{width:6,height:6,borderRadius:"50%",background:color,flexShrink:0,
        boxShadow:`0 0 6px ${color}`}}/>
      {children}
    </span>
  );
}

function Meter({ value, color, height=8, animated=true, C }) {
  return (
    <div style={{height,borderRadius:height,background:C.border,overflow:"hidden",width:"100%",flexShrink:0}}>
      <div style={{
        height:"100%", width:`${Math.min(value,100)}%`,
        background:`linear-gradient(90deg,${color}BB,${color})`,
        borderRadius:height,
        transition: animated ? "width 1s cubic-bezier(0.4,0,0.2,1)" : "none",
        boxShadow:`0 0 8px ${color}60`
      }}/>
    </div>
  );
}

function Spinner({ label="Processing…", C }) {
  return (
    <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:16,padding:"40px 0"}}>
      <div style={{position:"relative",width:56,height:56}}>
        <div style={{
          position:"absolute",inset:0,
          border:`3px solid ${C.border}`,borderRadius:"50%"
        }}/>
        <div style={{
          position:"absolute",inset:0,
          border:`3px solid transparent`,borderTopColor:C.accent,
          borderRadius:"50%", animation:"spin 0.7s linear infinite"
        }}/>
        <div style={{
          position:"absolute",inset:8,
          border:`2px solid transparent`,borderTopColor:C.accentHov,
          borderRadius:"50%", animation:"spin 1.1s linear infinite reverse"
        }}/>
      </div>
      <p style={{color:C.textSec,fontSize:13,fontWeight:500,animation:"pulse 1.5s ease infinite"}}>{label}</p>
    </div>
  );
}

function Card({ children, style={}, C, hover=false, glow=false }) {
  const [hov, setHov] = useState(false);
  return (
    <div
      onMouseEnter={()=>hover&&setHov(true)}
      onMouseLeave={()=>hover&&setHov(false)}
      style={{
        background:C.surface, border:`1px solid ${hov?C.borderHov:C.border}`,
        borderRadius:16, overflow:"hidden",
        boxShadow: glow&&hov ? `0 0 0 1px ${C.accent}40,${C.shadow}` : C.shadowSm,
        transition:"all 0.2s ease",
        ...style
      }}
    >{children}</div>
  );
}

function CardHeader({ title, sub, C, action }) {
  return (
    <div style={{
      padding:"14px 20px", borderBottom:`1px solid ${C.border}`,
      display:"flex", justifyContent:"space-between", alignItems:"center",
      background:C.surfaceHi
    }}>
      <div>
        <p style={{margin:0,fontSize:13,fontWeight:700,color:C.textPri}}>{title}</p>
        {sub && <p style={{margin:"2px 0 0",fontSize:11,color:C.textDim}}>{sub}</p>}
      </div>
      {action}
    </div>
  );
}

function StatBox({ label, value, color, sub, icon, C }) {
  return (
    <div style={{
      background:C.surfaceHi, border:`1px solid ${C.border}`,
      borderRadius:14, padding:"18px 16px", textAlign:"center",
      position:"relative", overflow:"hidden",
      boxShadow:C.shadowSm, transition:"transform 0.2s,box-shadow 0.2s"
    }}>
      <div style={{
        position:"absolute",top:-20,right:-20,width:80,height:80,
        borderRadius:"50%", background:color+"0D"
      }}/>
      {icon && <div style={{fontSize:22,marginBottom:8}}>{icon}</div>}
      <p style={{margin:0,fontSize:26,fontWeight:900,color:color||C.accent,
        textShadow:`0 0 20px ${color||C.accent}40`}}>{value}</p>
      <p style={{margin:"5px 0 2px",fontSize:11,fontWeight:600,color:C.textSec,
        textTransform:"uppercase",letterSpacing:"0.06em"}}>{label}</p>
      {sub && <p style={{margin:0,fontSize:11,color:C.textDim,fontWeight:500}}>{sub}</p>}
    </div>
  );
}

function Btn({ children, onClick, variant="primary", disabled=false, full=false, size="md", C }) {
  const [hov, setHov] = useState(false);
  const [act, setAct] = useState(false);
  const pad = size==="sm" ? "8px 16px" : size==="lg" ? "15px 28px" : "11px 22px";
  const fs  = size==="sm" ? 12 : size==="lg" ? 15 : 13;
  const bg  = variant==="primary"
    ? (act ? C.accentDim : hov ? C.accentHov : C.accent)
    : variant==="danger"
    ? (hov ? C.dangerDeep : C.danger)
    : variant==="ghost"
    ? (hov ? C.surfaceHov : "transparent")
    : (hov ? C.surfaceHov : C.surface);
  const border = variant==="outline" ? `1px solid ${hov?C.borderHov:C.border}`
    : variant==="ghost" ? "none"
    : "none";
  const color = variant==="primary" ? "#fff"
    : variant==="danger" ? "#fff"
    : C.textSec;
  return (
    <button
      onClick={onClick} disabled={disabled}
      onMouseEnter={()=>setHov(true)} onMouseLeave={()=>{setHov(false);setAct(false);}}
      onMouseDown={()=>setAct(true)} onMouseUp={()=>setAct(false)}
      style={{
        background:bg, border, borderRadius:10, color, padding:pad, fontSize:fs,
        fontWeight:700, cursor:disabled?"not-allowed":"pointer",
        width:full?"100%":"auto", opacity:disabled?0.5:1,
        transition:"all 0.15s", transform:act?"scale(0.97)":"scale(1)",
        boxShadow: variant==="primary"&&hov ? `0 4px 16px ${C.accent}50` : "none",
        display:"inline-flex", alignItems:"center", gap:8, justifyContent:"center",
        whiteSpace:"nowrap", letterSpacing:"0.01em"
      }}
    >{children}</button>
  );
}

function ThemeToggle({ dark, toggle, C }) {
  return (
    <button onClick={toggle} style={{
      background:C.surfaceHi, border:`1px solid ${C.border}`,
      borderRadius:24, padding:"5px 6px", cursor:"pointer",
      display:"flex", alignItems:"center", gap:4,
      transition:"all 0.3s", boxShadow:C.shadowSm
    }}>
      <div style={{
        width:28, height:28, borderRadius:20,
        background: dark ? "transparent" : C.accent,
        display:"flex", alignItems:"center", justifyContent:"center",
        transition:"all 0.3s", fontSize:15
      }}>☀️</div>
      <div style={{
        width:28, height:28, borderRadius:20,
        background: dark ? C.accent : "transparent",
        display:"flex", alignItems:"center", justifyContent:"center",
        transition:"all 0.3s", fontSize:15
      }}>🌙</div>
    </button>
  );
}

function chartTooltipStyle(C) {
  return {
    contentStyle:{
      background:C.surface, border:`1px solid ${C.border}`,
      borderRadius:10, fontSize:12, boxShadow:C.shadow, color:C.textPri
    },
    labelStyle:{color:C.textSec,fontWeight:600},
    itemStyle:{color:C.textPri}
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB 1 ─ Single Predict
// ═══════════════════════════════════════════════════════════════════════════════
function SinglePredict({ C }) {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [threshold, setThreshold] = useState(0.092);
  const [dragging, setDragging]   = useState(false);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f) return;
    setFile(f); setResult(null); setError(null);
    setPreview(URL.createObjectURL(f));
  };
  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0]; if (f) handleFile(f);
  }, []);

  const run = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    const form = new FormData();
    form.append("file", file);
    form.append("threshold", threshold);
    try {
      const res  = await fetch(`${API_URL}/predict`,{method:"POST",body:form});
      const data = await res.json();
      if (!res.ok) throw new Error(data.error||"Server error");
      setResult(data);
    } catch(e){ setError(e.message); }
    finally{ setLoading(false); }
  };

  const reset = () => { setFile(null); setPreview(null); setResult(null); setError(null); };
  const isPneu = result?.label === "PNEUMONIA";

  return (
    <div className="fade-in" style={{display:"flex",flexDirection:"column",gap:20}}>

      {/* Hero blurb */}
      {!preview && (
        <div style={{
          background:`linear-gradient(135deg,${C.accent}18,${C.purple}10)`,
          border:`1px solid ${C.accent}30`, borderRadius:16,
          padding:"28px 28px", position:"relative", overflow:"hidden"
        }}>
          <div style={{
            position:"absolute",top:-40,right:-40,width:160,height:160,
            borderRadius:"50%",background:C.accent+"0A"
          }}/>
          <p style={{fontSize:11,letterSpacing:"0.14em",color:C.accent,
            textTransform:"uppercase",fontWeight:700,marginBottom:8}}>
            🔬 Single Image Analysis
          </p>
          <h2 style={{fontSize:22,fontWeight:800,color:C.textPri,
            letterSpacing:"-0.02em",marginBottom:8,lineHeight:1.3}}>
            Upload a Chest X-Ray<br/>
            <span style={{color:C.accent}}>Get instant prediction</span>
          </h2>
          <p style={{fontSize:13,color:C.textSec,lineHeight:1.65,maxWidth:480}}>
            Drop a JPEG or PNG chest radiograph. The fine-tuned ConvNeXt-Tiny CNN
            analyses it in seconds and returns a pneumonia probability with confidence score.
          </p>
        </div>
      )}

      {/* Drop zone */}
      {!preview && (
        <div
          onClick={()=>inputRef.current.click()}
          onDrop={onDrop}
          onDragOver={(e)=>{e.preventDefault();setDragging(true);}}
          onDragLeave={()=>setDragging(false)}
          style={{
            border:`2px dashed ${dragging?C.accent:C.border}`,
            borderRadius:16, padding:"60px 32px", textAlign:"center",
            cursor:"pointer",
            background:dragging
              ? `linear-gradient(135deg,${C.accent}0F,${C.purple}08)`
              : C.surface,
            transition:"all 0.25s",
            boxShadow: dragging ? `0 0 0 4px ${C.accentGlow},${C.shadowSm}` : C.shadowSm
          }}
        >
          <div style={{
            width:72,height:72,borderRadius:20,
            background:`linear-gradient(135deg,${C.accent}20,${C.purple}15)`,
            border:`1px solid ${C.accent}30`,
            display:"flex",alignItems:"center",justifyContent:"center",
            margin:"0 auto 18px", fontSize:28,
            boxShadow: dragging?`0 0 24px ${C.accentGlow}`:"none",
            transition:"all 0.25s"
          }}>🫁</div>
          <p style={{color:C.textPri,fontWeight:700,fontSize:16,marginBottom:6}}>
            {dragging ? "Release to upload" : "Drop your chest X-ray here"}
          </p>
          <p style={{color:C.textDim,fontSize:13}}>
            or{" "}
            <span style={{color:C.accent,textDecoration:"underline",fontWeight:600}}>
              browse files
            </span>{" "}
            — JPG · PNG · BMP
          </p>
          <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.bmp"
            style={{display:"none"}} onChange={(e)=>handleFile(e.target.files[0])}/>
        </div>
      )}

      {/* Preview + result */}
      {preview && (
        <div className="fade-in" style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18}}>

          {/* Image card */}
          <Card C={C} glow>
            <CardHeader C={C} title="X-Ray Preview" sub={file?.name}
              action={
                <Btn variant="ghost" size="sm" C={C} onClick={reset}>✕ Remove</Btn>
              }/>
            <div style={{position:"relative",background:"#000",overflow:"hidden"}}>
              <img src={result?.preview||preview} alt="xray" style={{
                width:"100%", aspectRatio:"1", objectFit:"cover",
                display:"block", filter:"grayscale(100%) contrast(1.15) brightness(0.95)"
              }}/>
              {loading && (
                <div style={{
                  position:"absolute",inset:0,background:"#00000070",
                  display:"flex",alignItems:"center",justifyContent:"center"
                }}>
                  <div style={{
                    width:"100%",height:2,position:"absolute",top:"50%",
                    background:`linear-gradient(90deg,transparent,${C.accent},transparent)`,
                    animation:"scanline 1.5s ease-in-out infinite"
                  }}/>
                  <span style={{color:"#fff",fontSize:13,fontWeight:600,
                    background:"#00000080",padding:"6px 14px",borderRadius:8}}>
                    Analysing…
                  </span>
                </div>
              )}
              {result && (
                <div style={{
                  position:"absolute",top:10,right:10,
                }}>
                  <Badge color={isPneu?C.danger:C.safe} size="sm">
                    {result.label}
                  </Badge>
                </div>
              )}
            </div>
            <div style={{padding:"10px 16px",background:C.surfaceHi}}>
              <p style={{margin:0,fontSize:10,color:C.textDim,
                textTransform:"uppercase",letterSpacing:"0.06em"}}>
                {(file?.size/1024).toFixed(1)} KB · {file?.type||"image"}
              </p>
            </div>
          </Card>

          {/* Control + result card */}
          <Card C={C} style={{display:"flex",flexDirection:"column"}}>
            <CardHeader C={C} title="Analysis Controls" sub="Adjust threshold then run"/>
            <div style={{padding:20,display:"flex",flexDirection:"column",gap:18,flex:1}}>

              {/* Threshold */}
              <div style={{
                background:C.surfaceHi,borderRadius:12,padding:"14px 16px",
                border:`1px solid ${C.border}`
              }}>
                <div style={{display:"flex",justifyContent:"space-between",marginBottom:10}}>
                  <label style={{fontSize:11,color:C.textSec,fontWeight:700,
                    textTransform:"uppercase",letterSpacing:"0.08em"}}>
                    Threshold
                  </label>
                  <span style={{
                    fontSize:14,fontWeight:800,color:C.accent,
                    background:C.accent+"18",border:`1px solid ${C.accent}30`,
                    borderRadius:6,padding:"1px 10px"
                  }}>{threshold.toFixed(3)}</span>
                </div>
                <input type="range" min="0.05" max="0.95" step="0.005" value={threshold}
                  onChange={(e)=>setThreshold(parseFloat(e.target.value))}
                  style={{width:"100%",accentColor:C.accent,cursor:"pointer"}}/>
                <div style={{display:"flex",justifyContent:"space-between",marginTop:6}}>
                  <span style={{fontSize:10,color:C.textDim}}>← More sensitive</span>
                  <span style={{fontSize:10,color:C.textDim}}>More specific →</span>
                </div>
                <button onClick={()=>setThreshold(0.092)} style={{
                  marginTop:8,background:"none",border:"none",cursor:"pointer",
                  fontSize:10,color:C.accent,padding:0,textDecoration:"underline"
                }}>
                  Reset to validated optimum (0.092)
                </button>
              </div>

              {!loading && !result && !error && (
                <Btn variant="primary" full size="lg" C={C} onClick={run}>
                  <span>⚡</span> Analyse X-Ray
                </Btn>
              )}

              {loading && <Spinner label="Running ConvNeXt inference…" C={C}/>}

              {error && (
                <div className="fade-in" style={{
                  background:C.danger+"15",border:`1px solid ${C.danger}40`,
                  borderRadius:12,padding:16
                }}>
                  <p style={{margin:"0 0 8px",fontSize:12,fontWeight:700,color:C.danger}}>
                    ⚠ Error
                  </p>
                  <p style={{margin:"0 0 12px",fontSize:12,color:C.textSec}}>{error}</p>
                  <Btn variant="danger" size="sm" C={C} onClick={run}>Retry</Btn>
                </div>
              )}

              {result && !loading && (
                <div className="fade-in" style={{display:"flex",flexDirection:"column",gap:14}}>

                  {/* Big result */}
                  <div style={{
                    background: isPneu
                      ? `linear-gradient(135deg,${C.danger}18,${C.danger}08)`
                      : `linear-gradient(135deg,${C.safe}18,${C.safe}08)`,
                    border:`1px solid ${isPneu?C.danger:C.safe}40`,
                    borderRadius:14, padding:"18px 16px", textAlign:"center"
                  }}>
                    <div style={{fontSize:36,marginBottom:8}}>{isPneu?"🔴":"🟢"}</div>
                    <Badge color={isPneu?C.danger:C.safe} size="lg">{result.label}</Badge>
                    <p style={{margin:"10px 0 0",fontSize:28,fontWeight:900,
                      color:isPneu?C.danger:C.safe}}>
                      {result.confidence}%
                    </p>
                    <p style={{margin:2,fontSize:11,color:C.textDim}}>confidence</p>
                  </div>

                  {/* Meters */}
                  <div style={{display:"flex",flexDirection:"column",gap:10}}>
                    <div>
                      <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
                        <span style={{fontSize:11,color:C.textSec,fontWeight:500}}>Confidence</span>
                        <span style={{fontSize:12,fontWeight:700,color:isPneu?C.danger:C.safe}}>
                          {result.confidence}%
                        </span>
                      </div>
                      <Meter value={result.confidence} color={isPneu?C.danger:C.safe} C={C}/>
                    </div>
                    <div>
                      <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
                        <span style={{fontSize:11,color:C.textSec,fontWeight:500}}>
                          Pneumonia probability
                        </span>
                        <span style={{fontSize:12,fontWeight:700,color:C.accent}}>
                          {(result.probability*100).toFixed(1)}%
                        </span>
                      </div>
                      <Meter value={result.probability*100} color={C.accent} C={C}/>
                    </div>
                  </div>

                  {/* Clinical note */}
                  <div style={{
                    background: isPneu?C.danger+"10":C.safe+"10",
                    border:`1px solid ${isPneu?C.danger:C.safe}30`,
                    borderRadius:10, padding:"12px 14px",
                    fontSize:12, color:C.textSec, lineHeight:1.65
                  }}>
                    {isPneu
                      ? "⚠️ Pneumonia indicators detected. Model accuracy ~90% (threshold 0.5) — please confirm with a radiologist."
                      : "✓ No significant pneumonia markers. Model accuracy ~90% (threshold 0.5) — consult a physician regardless."}
                  </div>

                  <Btn variant="outline" full C={C} onClick={reset}>
                    ← Scan Another Image
                  </Btn>
                </div>
              )}
            </div>
          </Card>
        </div>
      )}

      {/* Stats strip */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:14}}>
        <StatBox label="Test Accuracy" value="90%"        color={C.accent}  icon="🎯" C={C}/>
        <StatBox label="Architecture" value="ConvNeXt"    color={C.safe}    icon="🧠" C={C}/>
        <StatBox label="Input Size"   value="224×224"     color={C.purple}  icon="📐" C={C}/>
      </div>

      <p style={{fontSize:11,color:C.textDim,textAlign:"center",lineHeight:1.7}}>
        For educational purposes only · ~90% test accuracy (threshold 0.5) ·{" "}
        <strong style={{color:C.textDim}}>Not a substitute for medical diagnosis</strong>
      </p>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB 2 ─ Batch Predict  (folder upload)
// ═══════════════════════════════════════════════════════════════════════════════
function BatchPredict({ C }) {
  const [files, setFiles]       = useState([]);
  const [results, setResults]   = useState([]);
  const [loading, setLoading]   = useState(false);
  const [progress, setProgress] = useState(0);
  const [activeChart, setActiveChart] = useState("bar");
  const [threshold, setThreshold]     = useState(0.092);
  const [dragging, setDragging]       = useState(false);
  const folderRef = useRef();
  const dropRef   = useRef();

  const handleFiles = (fileList) => {
    const arr = Array.from(fileList).filter(f =>
      /\.(jpe?g|png|bmp)$/i.test(f.name));
    setFiles(arr); setResults([]);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false);
    const items = e.dataTransfer.items;
    const out = [];
    if (items) {
      for (let item of items) {
        const entry = item.webkitGetAsEntry?.();
        if (entry?.isDirectory) {
          const reader = entry.createReader();
          reader.readEntries(entries => {
            entries.forEach(en => {
              if (en.isFile) en.file(f => {
                if (/\.(jpe?g|png|bmp)$/i.test(f.name)) out.push(f);
                if (out.length) setFiles(prev => [...prev, ...out]);
              });
            });
          });
          return;
        }
      }
    }
    // fallback: plain files
    handleFiles(e.dataTransfer.files);
  }, []);

  const runBatch = async () => {
    if (!files.length) return;
    setLoading(true); setResults([]); setProgress(0);
    const out = [];
    for (let i=0; i<files.length; i++) {
      const form = new FormData();
      form.append("file", files[i]);
      form.append("threshold", threshold);
      try {
        const res  = await fetch(`${API_URL}/predict`,{method:"POST",body:form});
        const data = await res.json();
        out.push({
          name:files[i].name, label:data.label||"ERROR",
          confidence:data.confidence||0, probability:data.probability||0,
          preview:data.preview||null, error:data.error||null
        });
      } catch(e) {
        out.push({name:files[i].name,label:"ERROR",confidence:0,probability:0,error:e.message});
      }
      setProgress(Math.round(((i+1)/files.length)*100));
      setResults([...out]);
    }
    setLoading(false);
  };

  const reset = () => { setFiles([]); setResults([]); setProgress(0); };

  const total   = results.length;
  const pneuCnt = results.filter(r=>r.label==="PNEUMONIA").length;
  const normCnt = results.filter(r=>r.label==="NORMAL").length;
  const errCnt  = results.filter(r=>r.label==="ERROR").length;
  const avgConf = total ? (results.reduce((s,r)=>s+r.confidence,0)/total).toFixed(1) : 0;

  const barData = [
    {name:"PNEUMONIA",count:pneuCnt,fill:C.danger},
    {name:"NORMAL",   count:normCnt,fill:C.safe},
    ...(errCnt?[{name:"ERROR",count:errCnt,fill:C.warn}]:[])
  ];
  const pieData = barData.map(d=>({name:d.name,value:d.count,fill:d.fill}));
  const lineData = results.map((r,i)=>({
    idx:i+1,
    prob:parseFloat((r.probability*100).toFixed(1)),
    label:r.label,
  }));
  const buckets = {"90–100":0,"70–90":0,"50–70":0,"<50":0};
  results.forEach(r=>{
    if(r.confidence>=90)      buckets["90–100"]++;
    else if(r.confidence>=70) buckets["70–90"]++;
    else if(r.confidence>=50) buckets["50–70"]++;
    else                      buckets["<50"]++;
  });
  const confData = Object.entries(buckets).map(([k,v])=>({range:k,count:v}));

  const CHART_TABS = [
    {id:"bar",   label:"Distribution", icon:"📊"},
    {id:"pie",   label:"Pie",          icon:"🥧"},
    {id:"line",  label:"Trend",        icon:"📈"},
    {id:"conf",  label:"Confidence",   icon:"🎯"},
  ];

  const exportCSV = () => {
    const csv = ["#,Filename,Label,Confidence,Probability",
      ...results.map((r,i)=>
        `${i+1},"${r.name}",${r.label},${r.confidence},${(r.probability*100).toFixed(1)}`)
    ].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
    a.download = "batch_results.csv"; a.click();
  };

  return (
    <div className="fade-in" style={{display:"flex",flexDirection:"column",gap:20}}>

      {/* Upload zone (folder drop or button) */}
      {!files.length && !results.length && (
        <>
          <div style={{
            background:`linear-gradient(135deg,${C.purple}18,${C.accent}10)`,
            border:`1px solid ${C.purple}30`,borderRadius:16,padding:"24px 28px"
          }}>
            <p style={{fontSize:11,letterSpacing:"0.14em",color:C.purple,
              textTransform:"uppercase",fontWeight:700,marginBottom:6}}>📂 Batch Analysis</p>
            <h2 style={{fontSize:22,fontWeight:800,color:C.textPri,marginBottom:6,
              letterSpacing:"-0.02em"}}>
              Upload a Folder of X-Rays
            </h2>
            <p style={{fontSize:13,color:C.textSec,lineHeight:1.65}}>
              Drag & drop an entire folder, or click the button to select a folder.
              All JPEG/PNG images inside will be analysed and results charted.
            </p>
          </div>

          <div
            ref={dropRef}
            onDrop={onDrop}
            onDragOver={(e)=>{e.preventDefault();setDragging(true);}}
            onDragLeave={()=>setDragging(false)}
            style={{
              border:`2px dashed ${dragging?C.purple:C.border}`,
              borderRadius:16, padding:"56px 32px", textAlign:"center",
              background: dragging
                ? `linear-gradient(135deg,${C.purple}0F,${C.accent}08)`
                : C.surface,
              transition:"all 0.25s",
              boxShadow: dragging ? `0 0 0 4px ${C.purple}25,${C.shadowSm}` : C.shadowSm
            }}
          >
            <div style={{
              width:72,height:72,borderRadius:20,
              background:`linear-gradient(135deg,${C.purple}25,${C.accent}18)`,
              border:`1px solid ${C.purple}35`,
              display:"flex",alignItems:"center",justifyContent:"center",
              margin:"0 auto 18px",fontSize:30
            }}>📁</div>
            <p style={{color:C.textPri,fontWeight:700,fontSize:16,marginBottom:6}}>
              {dragging ? "Release to load folder" : "Drag & drop a folder here"}
            </p>
            <p style={{color:C.textDim,fontSize:13,marginBottom:20}}>
              or click below to browse your folders
            </p>
            <Btn variant="primary" C={C} onClick={()=>folderRef.current.click()}>
              📂 Select Folder
            </Btn>
            {/* webkitdirectory = folder picker */}
            <input
              ref={folderRef} type="file"
              // @ts-ignore
              webkitdirectory="true" directory="true" multiple
              style={{display:"none"}}
              onChange={(e)=>handleFiles(e.target.files)}
            />
          </div>
        </>
      )}

      {/* File list + controls */}
      {files.length > 0 && !results.length && (
        <Card C={C} className="fade-in">
          <CardHeader C={C}
            title={`${files.length} images found`}
            sub="Ready for batch analysis"
            action={<Badge color={C.purple}>{files.length} files</Badge>}
          />
          <div style={{padding:18,display:"flex",flexDirection:"column",gap:16}}>

            {/* File preview list */}
            <div style={{
              maxHeight:200, overflowY:"auto",
              display:"flex", flexDirection:"column", gap:4,
              borderRadius:10, border:`1px solid ${C.border}`, overflow:"hidden"
            }}>
              {files.slice(0,200).map((f,i)=>(
                <div key={i} style={{
                  display:"flex", alignItems:"center", gap:10,
                  padding:"7px 14px",
                  background: i%2===0 ? C.surfaceHi : C.surface,
                  borderBottom:i<files.length-1?`1px solid ${C.border}`:"none"
                }}>
                  <span style={{fontSize:14}}>🖼️</span>
                  <span style={{fontSize:12,color:C.textSec,flex:1,
                    overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
                    {f.name}
                  </span>
                  <span style={{fontSize:10,color:C.textDim,whiteSpace:"nowrap"}}>
                    {(f.size/1024).toFixed(0)} KB
                  </span>
                </div>
              ))}
              {files.length>200 && (
                <div style={{padding:"8px 14px",background:C.surfaceHi,
                  textAlign:"center",fontSize:11,color:C.textDim}}>
                  …and {files.length-200} more
                </div>
              )}
            </div>

            {/* Threshold */}
            <div style={{
              background:C.surfaceHi,borderRadius:12,padding:"14px 16px",
              border:`1px solid ${C.border}`
            }}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:8}}>
                <label style={{fontSize:11,color:C.textSec,fontWeight:700,
                  textTransform:"uppercase",letterSpacing:"0.08em"}}>
                  Threshold
                </label>
                <span style={{fontSize:14,fontWeight:800,color:C.accent,
                  background:C.accent+"18",border:`1px solid ${C.accent}30`,
                  borderRadius:6,padding:"1px 10px"}}>{threshold.toFixed(3)}</span>
              </div>
              <input type="range" min="0.05" max="0.95" step="0.005" value={threshold}
                onChange={(e)=>setThreshold(parseFloat(e.target.value))}
                style={{width:"100%",accentColor:C.accent,cursor:"pointer"}}/>
              <div style={{display:"flex",justifyContent:"space-between",marginTop:4}}>
                <span style={{fontSize:10,color:C.textDim}}>← Sensitive</span>
                <span style={{fontSize:10,color:C.textDim}}>Specific →</span>
              </div>
            </div>

            <div style={{display:"flex",gap:10}}>
              <Btn variant="primary" full size="lg" C={C} onClick={runBatch}>
                ⚡ Run Batch Analysis
              </Btn>
              <Btn variant="outline" C={C} onClick={reset}>✕ Clear</Btn>
            </div>
          </div>
        </Card>
      )}

      {/* Live progress */}
      {loading && (
        <Card C={C} style={{padding:24}}>
          <div style={{display:"flex",justifyContent:"space-between",
            alignItems:"center",marginBottom:12}}>
            <p style={{fontSize:13,color:C.textSec,fontWeight:600,margin:0}}>
              Analysing image <strong style={{color:C.textPri}}>
                {results.length}
              </strong> of <strong style={{color:C.textPri}}>{files.length}</strong>
            </p>
            <span style={{fontSize:14,fontWeight:800,color:C.accent}}>{progress}%</span>
          </div>
          <Meter value={progress} color={C.accent} height={12} C={C}/>
          <div style={{
            marginTop:16,display:"flex",gap:12,flexWrap:"wrap"
          }}>
            {results.slice(-3).map((r,i)=>(
              <div key={i} style={{
                display:"flex",alignItems:"center",gap:6,
                background:C.surfaceHi,borderRadius:8,padding:"5px 10px"
              }}>
                <span style={{fontSize:10}}>
                  {r.label==="PNEUMONIA"?"🔴":"🟢"}
                </span>
                <span style={{fontSize:10,color:C.textSec,
                  maxWidth:120,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
                  {r.name}
                </span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Results */}
      {results.length > 0 && !loading && (
        <div className="fade-in" style={{display:"flex",flexDirection:"column",gap:18}}>

          {/* Summary cards */}
          <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12}}>
            <StatBox label="Total Scanned"   value={total}     color={C.accent}  icon="📋" C={C}/>
            <StatBox label="Pneumonia"       value={pneuCnt}
              sub={`${((pneuCnt/total)*100).toFixed(0)}%`}     color={C.danger}  icon="🔴" C={C}/>
            <StatBox label="Normal"          value={normCnt}
              sub={`${((normCnt/total)*100).toFixed(0)}%`}     color={C.safe}    icon="🟢" C={C}/>
            <StatBox label="Avg Confidence"  value={`${avgConf}%`} color={C.purple} icon="🎯" C={C}/>
          </div>

          {/* Charts */}
          <Card C={C}>
            <div style={{
              padding:"0 20px",borderBottom:`1px solid ${C.border}`,
              display:"flex",gap:0,background:C.surfaceHi,overflowX:"auto"
            }}>
              {CHART_TABS.map(t=>(
                <button key={t.id} onClick={()=>setActiveChart(t.id)} style={{
                  background:"none",border:"none",
                  borderBottom:`3px solid ${activeChart===t.id?C.accent:"transparent"}`,
                  color:activeChart===t.id?C.accent:C.textDim,
                  padding:"13px 18px",cursor:"pointer",fontSize:12,
                  fontWeight:activeChart===t.id?700:500,
                  display:"flex",alignItems:"center",gap:6,
                  transition:"all 0.15s", whiteSpace:"nowrap"
                }}>
                  {t.icon} {t.label}
                </button>
              ))}
            </div>
            <div style={{padding:20}}>
              {activeChart==="bar" && (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={barData} margin={{top:5,right:20,left:0,bottom:5}}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                    <XAxis dataKey="name" tick={{fill:C.textSec,fontSize:12}}/>
                    <YAxis tick={{fill:C.textSec,fontSize:11}}/>
                    <Tooltip {...chartTooltipStyle(C)}/>
                    <Bar dataKey="count" radius={[8,8,0,0]} maxBarSize={80}>
                      {barData.map((d,i)=><Cell key={i} fill={d.fill}/>)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
              {activeChart==="pie" && (
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie data={pieData} cx="50%" cy="50%"
                      innerRadius={60} outerRadius={110}
                      dataKey="value" paddingAngle={3}
                      label={({name,percent})=>`${name} ${(percent*100).toFixed(0)}%`}>
                      {pieData.map((d,i)=><Cell key={i} fill={d.fill}/>)}
                    </Pie>
                    <Tooltip {...chartTooltipStyle(C)}/>
                    <Legend wrapperStyle={{fontSize:12,color:C.textSec}}/>
                  </PieChart>
                </ResponsiveContainer>
              )}
              {activeChart==="line" && (
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={lineData} margin={{top:5,right:20,left:0,bottom:5}}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                    <XAxis dataKey="idx" tick={{fill:C.textSec,fontSize:11}}/>
                    <YAxis tick={{fill:C.textSec,fontSize:11}} domain={[0,100]}/>
                    <Tooltip {...chartTooltipStyle(C)}
                      formatter={(v)=>[`${v}%`,"Pneumonia prob"]}/>
                    <Line type="monotone" dataKey="prob" stroke={C.accent} strokeWidth={2}
                      dot={(p)=>{
                        const{cx,cy,payload}=p;
                        return <circle key={payload.idx} cx={cx} cy={cy} r={4}
                          fill={payload.label==="PNEUMONIA"?C.danger:C.safe} stroke="none"/>;
                      }}/>
                  </LineChart>
                </ResponsiveContainer>
              )}
              {activeChart==="conf" && (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={confData} margin={{top:5,right:20,left:0,bottom:5}}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                    <XAxis dataKey="range" tick={{fill:C.textSec,fontSize:12}}/>
                    <YAxis tick={{fill:C.textSec,fontSize:11}}/>
                    <Tooltip {...chartTooltipStyle(C)}/>
                    <Bar dataKey="count" fill={C.purple} radius={[8,8,0,0]} maxBarSize={70}/>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </Card>

          {/* Table */}
          <Card C={C}>
            <CardHeader C={C} title="Per-Image Results" sub={`${total} images analysed`}
              action={
                <Btn variant="primary" size="sm" C={C} onClick={exportCSV}>
                  ↓ Export CSV
                </Btn>
              }/>
            <div style={{maxHeight:300,overflowY:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
                <thead style={{position:"sticky",top:0,zIndex:1}}>
                  <tr style={{background:C.surfaceHi}}>
                    {["#","Filename","Label","Confidence","Probability"].map(h=>(
                      <th key={h} style={{
                        padding:"10px 16px",textAlign:"left",
                        color:C.textDim,fontWeight:700,fontSize:10,
                        letterSpacing:"0.07em",textTransform:"uppercase",
                        borderBottom:`1px solid ${C.border}`,whiteSpace:"nowrap"
                      }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {results.map((r,i)=>(
                    <tr key={i} style={{
                      borderBottom:`1px solid ${C.border}`,
                      background:i%2===0?C.surface:C.surfaceHi,
                      transition:"background 0.15s"
                    }}>
                      <td style={{padding:"9px 16px",color:C.textDim,fontWeight:600}}>{i+1}</td>
                      <td style={{padding:"9px 16px",color:C.textSec,
                        maxWidth:160,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
                        {r.name}
                      </td>
                      <td style={{padding:"9px 16px"}}>
                        <Badge color={r.label==="PNEUMONIA"?C.danger:r.label==="ERROR"?C.warn:C.safe}>
                          {r.label}
                        </Badge>
                      </td>
                      <td style={{padding:"9px 16px"}}>
                        <div style={{display:"flex",alignItems:"center",gap:8,minWidth:110}}>
                          <div style={{flex:1}}>
                            <Meter value={r.confidence}
                              color={r.label==="PNEUMONIA"?C.danger:C.safe} height={5} C={C}/>
                          </div>
                          <span style={{color:C.textPri,fontWeight:700,fontSize:11,minWidth:34,
                            textAlign:"right"}}>{r.confidence}%</span>
                        </div>
                      </td>
                      <td style={{padding:"9px 16px",color:C.textSec,fontWeight:600}}>
                        {(r.probability*100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <div style={{display:"flex",gap:12}}>
            <Btn variant="outline" full C={C} onClick={reset}>← New Batch</Btn>
            <Btn variant="primary" C={C} onClick={exportCSV}>↓ Export CSV</Btn>
          </div>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB 3 ─ Model Metrics
// ═══════════════════════════════════════════════════════════════════════════════
function ModelMetrics({ C }) {
  const epochs  = TRAIN_HISTORY.loss.map((_,i)=>i+1);
  const lossData = epochs.map((e,i)=>({
    epoch:e,
    "Train Loss": parseFloat(TRAIN_HISTORY.loss[i].toFixed(3)),
    "Val Loss":   parseFloat(TRAIN_HISTORY.val_loss[i].toFixed(3)),
  }));
  const accData = epochs.map((e,i)=>({
    epoch:e,
    "Train Acc": parseFloat((TRAIN_HISTORY.acc[i]*100).toFixed(1)),
    "Val Acc":   parseFloat((TRAIN_HISTORY.val_acc[i]*100).toFixed(1)),
  }));

  const tot  = CM.TP+CM.FP+CM.FN+CM.TN;
  const acc  = ((CM.TP+CM.TN)/tot*100).toFixed(1);
  const prec = (CM.TP/(CM.TP+CM.FP)*100).toFixed(1);
  const rec  = (CM.TP/(CM.TP+CM.FN)*100).toFixed(1);
  const f1   = (2*CM.TP/(2*CM.TP+CM.FP+CM.FN)*100).toFixed(1);
  const spec = (CM.TN/(CM.TN+CM.FP)*100).toFixed(1);

  return (
    <div className="fade-in" style={{display:"flex",flexDirection:"column",gap:20}}>

      {/* Metrics strip */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:12}}>
        <StatBox label="Accuracy"    value={`${acc}%`}  color={C.accent}  icon="🎯" C={C}/>
        <StatBox label="Precision"   value={`${prec}%`} color={C.purple}  icon="🔍" C={C}/>
        <StatBox label="Recall"      value={`${rec}%`}  color={C.safe}    icon="📡" C={C}/>
        <StatBox label="F1 Score"    value={`${f1}%`}   color={C.warn}    icon="⚖️" C={C}/>
        <StatBox label="Specificity" value={`${spec}%`} color={C.accent}  icon="🛡️" C={C}/>
      </div>

      {/* Training curves */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
        <Card C={C}>
          <CardHeader C={C} title="📉 Loss Curve" sub="Train vs Validation (Phase 1: epochs 1-5 head only, Phase 2: 6-7 CNN fine-tune)"/>
          <div style={{padding:"16px 10px 10px"}}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                <XAxis dataKey="epoch" tick={{fill:C.textSec,fontSize:11}}/>
                <YAxis tick={{fill:C.textSec,fontSize:11}} domain={[0, 0.5]}/>
                <Tooltip {...chartTooltipStyle(C)}/>
                <Legend wrapperStyle={{fontSize:11,color:C.textSec}}/>
                <ReferenceLine x={PHASE1_EPOCHS + 0.5} stroke={C.warn} strokeDasharray="2 2"
                  label={{value:"Phase 1→2", position:"top", fill:C.warn, fontSize:10}}/>
                <Line type="monotone" dataKey="Train Loss" stroke={C.accent}
                  strokeWidth={2.5} dot={false}/>
                <Line type="monotone" dataKey="Val Loss" stroke={C.danger}
                  strokeWidth={2.5} dot={false} strokeDasharray="6 3"/>
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card C={C}>
          <CardHeader C={C} title="📈 Accuracy Curve" sub="Train vs Validation (Phase 1: epochs 1-5 head only, Phase 2: 6-7 CNN fine-tune)"/>
          <div style={{padding:"16px 10px 10px"}}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={accData}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                <XAxis dataKey="epoch" tick={{fill:C.textSec,fontSize:11}}/>
                <YAxis tick={{fill:C.textSec,fontSize:11}} domain={[60, 100]}/>
                <Tooltip {...chartTooltipStyle(C)}/>
                <Legend wrapperStyle={{fontSize:11,color:C.textSec}}/>
                <ReferenceLine x={PHASE1_EPOCHS + 0.5} stroke={C.warn} strokeDasharray="2 2"
                  label={{value:"Phase 1→2", position:"top", fill:C.warn, fontSize:10}}/>
                <Line type="monotone" dataKey="Train Acc" stroke={C.safe}
                  strokeWidth={2.5} dot={false}/>
                <Line type="monotone" dataKey="Val Acc" stroke={C.purple}
                  strokeWidth={2.5} dot={false} strokeDasharray="6 3"/>
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Confusion matrix */}
      <Card C={C}>
        <CardHeader C={C} title="🔢 Confusion Matrix" sub="At threshold=0.5 — 624 test images (Acc 90%). Val-tuned threshold=0.092 gives Acc 86%, Recall 99%"/>
        <div style={{padding:18,display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
          {[
            {label:"True Positive",  val:CM.TP, color:C.safe,   icon:"✅", desc:"Pneumonia correctly detected"},
            {label:"False Positive", val:CM.FP, color:C.warn,   icon:"⚠️", desc:"Normal misclassified as Pneumonia"},
            {label:"False Negative", val:CM.FN, color:C.danger, icon:"❌", desc:"Pneumonia missed by model"},
            {label:"True Negative",  val:CM.TN, color:C.accent, icon:"✅", desc:"Normal correctly identified"},
          ].map(c=>(
            <div key={c.label} style={{
              background:`linear-gradient(135deg,${c.color}12,${c.color}06)`,
              border:`1px solid ${c.color}35`,borderRadius:14,
              padding:"18px 16px",display:"flex",flexDirection:"column",gap:6
            }}>
              <span style={{fontSize:22}}>{c.icon}</span>
              <span style={{fontSize:32,fontWeight:900,color:c.color,
                textShadow:`0 0 20px ${c.color}40`}}>{c.val}</span>
              <span style={{fontSize:12,fontWeight:700,color:C.textPri}}>{c.label}</span>
              <span style={{fontSize:11,color:C.textDim,lineHeight:1.5}}>{c.desc}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Architecture */}
      <Card C={C}>
        <CardHeader C={C} title="🧠 ConvNeXt-Tiny Architecture" sub="torchvision ImageNet-1k pretrained, 2-phase fine-tuning"/>
        <div style={{padding:16,display:"flex",flexDirection:"column",gap:8}}>
          {[
            {block:"Backbone", detail:"ConvNeXt-Tiny — 4 stages of depthwise conv blocks, GELU, LayerNorm (28M params total)", color:C.accent},
            {block:"Phase 1",  detail:"Backbone frozen → train classifier head only (5 epochs, lr=1e-3)", color:C.purple},
            {block:"Phase 2",  detail:"Freeze all → unfreeze features.6 + features.7 + classifier (2 epochs, backbone lr=5e-6)", color:C.safe},
            {block:"Pool",     detail:"AdaptiveAvgPool → LayerNorm2d → Flatten  (from ConvNeXt backbone)", color:C.warn},
            {block:"Head",     detail:"Linear(768→256) → GELU → Drop(0.4) → Linear(256→64) → GELU → Drop(0.3) → Linear(64→1)", color:C.danger},
          ].map((b,i)=>(
            <div key={i} style={{
              display:"flex",alignItems:"center",gap:14,
              padding:"11px 16px",background:C.surfaceHi,borderRadius:10,
              border:`1px solid ${C.border}`
            }}>
              <div style={{
                minWidth:70,fontSize:11,fontWeight:800,color:b.color,
                letterSpacing:"0.03em",
                background:b.color+"15",borderRadius:6,
                padding:"3px 8px",textAlign:"center"
              }}>{b.block}</div>
              <span style={{fontSize:12,color:C.textSec,fontFamily:"'Courier New',monospace"}}>
                {b.detail}
              </span>
            </div>
          ))}
        </div>
      </Card>


    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB 4 ─ About
// ═══════════════════════════════════════════════════════════════════════════════
function About({ C }) {
  const steps = [
    {icon:"📤",title:"Upload X-Ray",     desc:"JPEG or PNG chest radiograph via drag-and-drop, file browser, or folder upload."},
    {icon:"🔧",title:"Preprocessing",    desc:"Resized to 224×224px, normalised with ImageNet mean/std [0.485,0.456,0.406] / [0.229,0.224,0.225], augmented with rotation, color jitter & flips during training."},
    {icon:"🧠",title:"ConvNeXt Inference",desc:"28M-param ConvNeXt-Tiny backbone — 4 stages of large-kernel depthwise convolutions with GELU, LayerNorm, and inverted-bottleneck blocks."},
    {icon:"📊",title:"Sigmoid Output",   desc:"Custom 3-layer head outputs pneumonia probability [0,1]. Default threshold 0.092 (val-tuned max-F1) catches 99.2% of pneumonia cases."},
    {icon:"⚖️",title:"Class Weighting", desc:"Imbalanced dataset (3875 vs 1341) handled via pos_weight=0.346 in BCEWithLogitsLoss — upweights normal class during training."},
    {icon:"🩺",title:"Clinical Note",    desc:"~90% test accuracy at threshold 0.5. All results must be reviewed by a qualified radiologist — not for diagnosis."},
  ];
  const techStack = [
    {name:"PyTorch",          role:"Model training & inference", icon:"🔥"},
    {name:"TorchVision",      role:"ConvNeXt-Tiny pretrained",   icon:"👁️"},
    {name:"Flask",            role:"REST API backend",           icon:"🌐"},
    {name:"Flask-CORS",       role:"Cross-origin requests",      icon:"🔗"},
    {name:"React 18",         role:"Frontend UI",                icon:"⚛️"},
    {name:"Recharts",         role:"Data visualisation",         icon:"📊"},
  ];
  return (
    <div className="fade-in" style={{display:"flex",flexDirection:"column",gap:20}}>

      {/* Dataset */}
      <Card C={C}>
        <CardHeader C={C} title="📋 Dataset Summary" sub="Chest X-Ray Pneumonia — Kaggle"/>
        <div style={{padding:16,display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12}}>
          {[
            {split:"Train",normal:1342,pneu:3876},
            {split:"Test", normal:234, pneu:390},
            {split:"Val",  normal:9,   pneu:8},
          ].map(d=>(
            <div key={d.split} style={{
              background:C.surfaceHi,borderRadius:12,padding:16,
              border:`1px solid ${C.border}`
            }}>
              <p style={{margin:"0 0 12px",fontSize:14,fontWeight:800,color:C.textPri}}>
                {d.split}
              </p>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:6}}>
                <span style={{fontSize:12,color:C.safe,fontWeight:600}}>● Normal</span>
                <span style={{fontSize:12,color:C.textPri,fontWeight:700}}>{d.normal}</span>
              </div>
              <div style={{display:"flex",justifyContent:"space-between"}}>
                <span style={{fontSize:12,color:C.danger,fontWeight:600}}>● Pneumonia</span>
                <span style={{fontSize:12,color:C.textPri,fontWeight:700}}>{d.pneu}</span>
              </div>
              <div style={{marginTop:10}}>
                <Meter
                  value={d.pneu/(d.normal+d.pneu)*100}
                  color={C.danger} height={4} C={C}/>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Pipeline */}
      <Card C={C}>
        <CardHeader C={C} title="⚙️ How It Works" sub="End-to-end pipeline"/>
        <div style={{padding:16,display:"flex",flexDirection:"column",gap:10}}>
          {steps.map((s,i)=>(
            <div key={i} style={{
              display:"flex",gap:14,padding:"13px 16px",
              background:C.surfaceHi,borderRadius:12,
              border:`1px solid ${C.border}`,alignItems:"flex-start"
            }}>
              <div style={{
                fontSize:20,width:40,height:40,borderRadius:10,
                background:C.surface,display:"flex",alignItems:"center",
                justifyContent:"center",flexShrink:0,border:`1px solid ${C.border}`
              }}>{s.icon}</div>
              <div>
                <p style={{margin:"0 0 3px",fontSize:13,fontWeight:700,color:C.textPri}}>
                  {s.title}
                </p>
                <p style={{margin:0,fontSize:12,color:C.textSec,lineHeight:1.65}}>{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Tech stack */}
      <Card C={C}>
        <CardHeader C={C} title="🛠️ Tech Stack"/>
        <div style={{padding:16,display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
          {techStack.map(t=>(
            <div key={t.name} style={{
              display:"flex",alignItems:"center",gap:12,
              padding:"11px 14px",background:C.surfaceHi,borderRadius:10,
              border:`1px solid ${C.border}`
            }}>
              <span style={{fontSize:18}}>{t.icon}</span>
              <div>
                <p style={{margin:0,fontSize:12,fontWeight:700,color:C.textPri}}>{t.name}</p>
                <p style={{margin:0,fontSize:10,color:C.textDim}}>{t.role}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROOT
// ═══════════════════════════════════════════════════════════════════════════════
const TABS = [
  {id:"single",  label:"Single Predict", icon:"🔬"},
  {id:"batch",   label:"Batch Predict",  icon:"📂"},
  {id:"metrics", label:"Model Metrics",  icon:"📈"},
  {id:"about",   label:"About",          icon:"ℹ️"},
];

export default function App() {
  const { C, dark, toggle } = useTheme();
  const [tab, setTab] = useState("single");

  return (
    <>
      <style>{GLOBAL_CSS}</style>
      <div style={{
        minHeight:"100vh",
        background:C.bgGrad,
        color:C.textPri,
        fontFamily:"'Inter','Segoe UI',sans-serif",
        display:"flex", flexDirection:"column",
        transition:"background 0.3s,color 0.3s"
      }}>

        {/* ── Header ── */}
        <header style={{
          background:C.glass,
          backdropFilter:"blur(20px)",
          WebkitBackdropFilter:"blur(20px)",
          borderBottom:`1px solid ${C.border}`,
          padding:"0 28px", height:60,
          display:"flex", alignItems:"center", justifyContent:"space-between",
          position:"sticky", top:0, zIndex:200,
          boxShadow:C.shadowSm
        }}>
          <div style={{display:"flex",alignItems:"center",gap:12}}>
            <div style={{
              width:36,height:36,borderRadius:10,
              background:`linear-gradient(135deg,${C.accent},${C.purple})`,
              display:"flex",alignItems:"center",justifyContent:"center",
              fontSize:18, boxShadow:`0 4px 12px ${C.accent}40`
            }}>🫁</div>
            <div>
              <span style={{fontWeight:900,fontSize:16,letterSpacing:"-0.02em",color:C.textPri}}>
                PneumoScan
              </span>
              <span style={{fontWeight:900,fontSize:16,letterSpacing:"-0.02em",
                background:`linear-gradient(90deg,${C.accent},${C.purple})`,
                WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
                AI
              </span>
            </div>
          </div>

          <div style={{display:"flex",alignItems:"center",gap:10}}>
            <span style={{
              fontSize:10,color:C.textDim,border:`1px solid ${C.border}`,
              borderRadius:6,padding:"3px 9px",letterSpacing:"0.06em",
              background:C.surfaceHi
            }}>ConvNeXt</span>
            <span style={{
              fontSize:10,
              background:C.safe+"22",color:C.safe,
              border:`1px solid ${C.safe}44`,borderRadius:6,
              padding:"3px 9px",letterSpacing:"0.06em",fontWeight:600
            }}>~90% TEST ACC</span>
            <ThemeToggle dark={dark} toggle={toggle} C={C}/>
          </div>
        </header>

        {/* ── Tab bar ── */}
        <div style={{
          background:C.surface,
          borderBottom:`1px solid ${C.border}`,
          padding:"0 28px", display:"flex", gap:0,
          overflowX:"auto",
          boxShadow:`0 1px 0 ${C.border}`
        }}>
          {TABS.map(t=>(
            <button key={t.id} onClick={()=>setTab(t.id)} style={{
              background:"none", border:"none",
              borderBottom:`3px solid ${tab===t.id ? C.accent : "transparent"}`,
              color: tab===t.id ? C.accent : C.textDim,
              padding:"14px 20px", cursor:"pointer", fontSize:13,
              fontWeight: tab===t.id ? 700 : 500,
              display:"flex", alignItems:"center", gap:8,
              transition:"all 0.15s", whiteSpace:"nowrap",
              outline:"none"
            }}>
              <span style={{fontSize:15}}>{t.icon}</span>{t.label}
            </button>
          ))}
        </div>

        {/* ── Content ── */}
        <main style={{
          flex:1, maxWidth:900, width:"100%",
          margin:"0 auto", padding:"28px 24px 64px",
          boxSizing:"border-box"
        }}>
          {tab==="single"  && <SinglePredict C={C}/>}
          {tab==="batch"   && <BatchPredict  C={C}/>}
          {tab==="metrics" && <ModelMetrics  C={C}/>}
          {tab==="about"   && <About         C={C}/>}
        </main>

        {/* ── Footer ── */}
        <footer style={{
          borderTop:`1px solid ${C.border}`,
          background:C.surface,
          padding:"14px 28px",
          display:"flex", alignItems:"center", justifyContent:"space-between"
        }}>
          <span style={{fontSize:11,color:C.textDim}}>
            PneumoScanAI · Educational use only · Not for clinical diagnosis
          </span>
          <span style={{fontSize:11,color:C.textDim}}>
            ConvNeXt-Tiny · ~90% Test Accuracy · Val-tuned Threshold 0.092
          </span>
        </footer>
      </div>
    </>
  );
}