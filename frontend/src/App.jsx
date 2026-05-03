import { useState, useCallback, useRef } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8080";

// ─── Helpers ──────────────────────────────────────────────────────────────────
const cls = (...a) => a.filter(Boolean).join(" ");

function ConfidenceBar({ label, value, color }) {
  return (
    <div className="prob-row">
      <span className="prob-label">{label}</span>
      <div className="prob-track">
        <div
          className="prob-fill"
          style={{ width: `${(value * 100).toFixed(1)}%`, background: color }}
        />
      </div>
      <span className="prob-value">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [file, setFile]           = useState(null);
  const [preview, setPreview]     = useState(null);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [dragging, setDragging]   = useState(false);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(f));
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  }, []);

  const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const { data } = await axios.post(`${API_BASE}/predict`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 60000,
      });
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null); setPreview(null); setResult(null); setError(null);
  };

  const isPneumonia = result?.prediction === "PNEUMONIA";

  return (
    <div className="app">
      {/* ── Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">⬡</span>
            <span className="logo-text">PneumoScan<span className="logo-ai">AI</span></span>
          </div>
          <p className="header-sub">
            Chest X-Ray Analysis · ResNet-18 · NVIDIA Triton · GPT-4
          </p>
        </div>
      </header>

      <main className="main">
        {/* ── Upload Zone */}
        <section className="upload-section">
          <div
            className={cls("drop-zone", dragging && "dragging", preview && "has-preview")}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => !preview && inputRef.current.click()}
          >
            {preview ? (
              <div className="preview-wrap">
                <img src={preview} alt="X-Ray preview" className="preview-img" />
                <button className="remove-btn" onClick={(e) => { e.stopPropagation(); reset(); }}>
                  ✕ Remove
                </button>
              </div>
            ) : (
              <div className="drop-prompt">
                <div className="drop-icon">⬆</div>
                <p className="drop-title">Drop your chest X-ray here</p>
                <p className="drop-sub">or click to browse · JPEG / PNG · max 20 MB</p>
              </div>
            )}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files?.[0])}
          />

          <button
            className={cls("analyze-btn", loading && "loading", !file && "disabled")}
            onClick={analyze}
            disabled={!file || loading}
          >
            {loading ? (
              <span className="spinner-wrap"><span className="spinner" /> Analysing…</span>
            ) : "Run Analysis"}
          </button>

          {error && (
            <div className="error-box">
              <span className="error-icon">⚠</span> {error}
            </div>
          )}
        </section>

        {/* ── Results */}
        {result && (
          <section className="results-section">
            <div className={cls("verdict-card", isPneumonia ? "pneumonia" : "normal")}>
              <div className="verdict-badge">
                {isPneumonia ? "⚠" : "✓"}
              </div>
              <div className="verdict-text">
                <h2 className="verdict-label">{result.prediction}</h2>
                <p className="verdict-conf">
                  {(result.confidence * 100).toFixed(1)}% confidence
                </p>
              </div>
              <div className="latency-tag">
                ⚡ {result.inference_time_ms.toFixed(0)} ms
              </div>
            </div>

            {/* Probabilities */}
            <div className="card probs-card">
              <h3 className="card-title">Class Probabilities</h3>
              <ConfidenceBar
                label="NORMAL"
                value={result.probabilities.NORMAL}
                color="var(--green)"
              />
              <ConfidenceBar
                label="PNEUMONIA"
                value={result.probabilities.PNEUMONIA}
                color="var(--red)"
              />
            </div>

            {/* LLM Report */}
            <div className="card report-card">
              <h3 className="card-title">
                <span className="report-icon">✦</span> AI Diagnostic Report
                <span className="gpt-badge">GPT-4o</span>
              </h3>
              <div className="report-body">
                {result.report.split("\n\n").map((para, i) => (
                  <p key={i} className="report-para">{para}</p>
                ))}
              </div>
              <p className="disclaimer">
                ⚠ This AI analysis is for research purposes only and does not
                constitute medical advice. Always consult a qualified radiologist.
              </p>
            </div>

            <button className="reset-btn" onClick={reset}>
              ← Analyse another image
            </button>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>PneumoScan AI · ResNet-18 + NVIDIA Triton + GPT-4o · Built for research</p>
      </footer>
    </div>
  );
}
