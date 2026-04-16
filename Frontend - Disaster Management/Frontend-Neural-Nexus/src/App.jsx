import { useEffect, useMemo, useState } from 'react';
import { mockClaims } from './data/mockClaims';
import { buildReport, deriveInsights } from './utils/report';

const initialForm = {
  claimId: '',
  location: '',
  incidentType: '',
};

const incidentOptions = ['Flood', 'Fire', 'Earthquake', 'Cyclone', 'Landslide'];

const severityTone = {
  low: 'tone-green',
  medium: 'tone-amber',
  high: 'tone-red',
};

const authenticityTone = {
  'Likely Genuine': 'tone-green',
  'Possibly Exaggerated': 'tone-amber',
  'Suspicious Claim': 'tone-red',
};

function App() {
  const [formData, setFormData] = useState(initialForm);
  const [previewUrl, setPreviewUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [reportText, setReportText] = useState('');
  const [error, setError] = useState('');

  const insights = useMemo(() => (analysis ? deriveInsights(analysis) : null), [analysis]);

  useEffect(() => () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
  }, [previewUrl]);

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormData((current) => ({ ...current, [name]: value }));
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setAnalysis(null);
    setReportText('');
    setError('');
  };

  const handleAnalyze = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError('Upload an image before running analysis.');
      return;
    }

    const payload = new FormData();
    payload.append('image', selectedFile);
    payload.append('claimId', formData.claimId);
    payload.append('location', formData.location);
    payload.append('incidentType', formData.incidentType);

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: payload,
      });

      if (!response.ok) {
        throw new Error('Analysis request failed.');
      }

      const result = await response.json();
      setAnalysis(result);
    } catch (requestError) {
      setError(requestError.message || 'Unable to analyze this file right now.');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = () => {
    if (!analysis || !insights) return;
    const nextReport = buildReport({ formData, result: analysis, insights });
    setReportText(nextReport);
  };

  const handleDownloadReport = () => {
    if (!reportText) return;
    const blob = new Blob([reportText], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${formData.claimId || 'claim-report'}.txt`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  return (
    <div className="shell">
      <aside className="sidebar glass">
        <div>
          <p className="eyebrow">Operations Console</p>
          <h2>Claims Command Center</h2>
          <p className="muted">
            Monitor incoming cases, prioritize severe events, and escalate suspicious claims.
          </p>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-heading">
            <span>Historical Claims</span>
            <span className="pill neutral">Live</span>
          </div>
          <div className="claim-list">
            {mockClaims.map((claim) => (
              <div className="claim-item" key={claim.id}>
                <div>
                  <strong>{claim.id}</strong>
                  <p>{claim.location}</p>
                </div>
                <div className="claim-meta">
                  <span className="pill neutral">{claim.type}</span>
                  <small>{claim.status}</small>
                  <strong>{claim.payout}</strong>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="metric-stack">
          <div className="mini-card">
            <span>Claims Processed Today</span>
            <strong>148</strong>
          </div>
          <div className="mini-card">
            <span>Avg. AI Turnaround</span>
            <strong>17 sec</strong>
          </div>
          <div className="mini-card">
            <span>Manual Audits Triggered</span>
            <strong>12</strong>
          </div>
        </div>
      </aside>

      <main className="content">
        <section className="hero">
          <div>
            <p className="eyebrow">Disaster Damage Intelligence System</p>
            <h1>AI-Powered Insurance Claim Verification</h1>
            <p className="hero-copy">
              A modern insurer workflow for validating disaster claims with fast visual analysis,
              severity scoring, fraud indicators, and structured reporting.
            </p>
          </div>
          <div className="hero-panel glass">
            <div className="hero-stat">
              <span>AI Confidence Pipeline</span>
              <strong>98.2%</strong>
            </div>
            <div className="hero-stat">
              <span>Severe Claims Flagged</span>
              <strong>24</strong>
            </div>
            <div className="hero-stat">
              <span>Potential Fraud Alerts</span>
              <strong>5</strong>
            </div>
          </div>
        </section>

        <section className="grid">
          <article className="card glass upload-card">
            <div className="card-header">
              <div>
                <p className="section-label">Upload Claim Evidence</p>
                <h3>Submit field evidence for AI review</h3>
              </div>
              <span className="pill neutral">Image-ready</span>
            </div>

            <form className="upload-form" onSubmit={handleAnalyze}>
              <label className="dropzone">
                <input type="file" accept="image/*" onChange={handleFileChange} />
                <div>
                  <strong>{selectedFile ? selectedFile.name : 'Drop an image or click to browse'}</strong>
                  <p>JPG, PNG, WEBP supported. Video can be added later without changing the UI pattern.</p>
                </div>
              </label>

              <div className="form-grid">
                <label>
                  Claim ID
                  <input
                    name="claimId"
                    value={formData.claimId}
                    onChange={handleInputChange}
                    placeholder="CLM-2026-1042"
                  />
                </label>
                <label>
                  Location
                  <input
                    name="location"
                    value={formData.location}
                    onChange={handleInputChange}
                    placeholder="Kolkata, West Bengal"
                  />
                </label>
                <label className="full-width">
                  Incident Type
                  <select name="incidentType" value={formData.incidentType} onChange={handleInputChange}>
                    <option value="">Auto-detect from image</option>
                    {incidentOptions.map((option) => (
                      <option value={option} key={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="action-row">
                <button className="primary-button" type="submit" disabled={loading}>
                  {loading ? 'Analyzing disaster damage...' : 'Analyze Damage'}
                </button>
                {error ? <p className="error-text">{error}</p> : null}
              </div>
            </form>
          </article>

          <article className="card glass analysis-card">
            <div className="card-header">
              <div>
                <p className="section-label">View AI Analysis</p>
                <h3>Claim assessment at a glance</h3>
              </div>
            </div>

            <div className="analysis-layout">
              <div className="preview-panel">
                {previewUrl ? (
                  <img src={previewUrl} alt="Uploaded claim evidence" className="preview-image" />
                ) : (
                  <div className="preview-placeholder">
                    <span>Before / After Placeholder</span>
                    <p>Upload claim evidence to preview the impacted asset.</p>
                  </div>
                )}
              </div>

              <div className="analysis-summary">
                <div className="summary-row">
                  <span>Disaster Type</span>
                  <strong>{analysis?.disasterType || 'Awaiting analysis'}</strong>
                </div>
                <div className="summary-row">
                  <span>Damage Severity</span>
                  <strong className={`pill ${severityTone[(analysis?.damageSeverity || '').toLowerCase()] || 'neutral'}`}>
                    {analysis?.damageSeverity || 'N/A'}
                  </strong>
                </div>
                <div className="summary-row">
                  <span>Confidence Score</span>
                  <strong>{analysis ? `${Math.round((analysis.confidenceScore || 0) * 100)}%` : '--'}</strong>
                </div>
                <div className="summary-row metadata">
                  <span>Metadata</span>
                  <pre>{analysis ? JSON.stringify(analysis.metadata || {}, null, 2) : 'Run analysis to view returned metadata.'}</pre>
                </div>
              </div>
            </div>
          </article>
        </section>

        <section className="grid lower-grid">
          <article className="card glass">
            <div className="card-header">
              <div>
                <p className="section-label">Damage Insights Panel</p>
                <h3>Insurance-facing severity and exposure metrics</h3>
              </div>
            </div>
            <div className="insights-grid">
              <div className="insight-card">
                <span>Estimated Damage</span>
                <strong>{insights ? `${insights.damagePct}%` : '--'}</strong>
              </div>
              <div className="insight-card">
                <span>Estimated Loss</span>
                <strong>{insights?.estimatedLoss || '--'}</strong>
              </div>
              <div className="insight-card">
                <span>Risk Score</span>
                <strong>{insights?.riskScore || '--'}</strong>
              </div>
            </div>
          </article>

          <article className="card glass authenticity-card">
            <div className="card-header">
              <div>
                <p className="section-label">Claim Authenticity Check</p>
                <h3>Fraud detection indicator</h3>
              </div>
            </div>
            <div className={`authenticity-banner ${authenticityTone[insights?.authenticity] || 'neutral'}`}>
              <span className="auth-icon">
                {insights?.authenticity === 'Likely Genuine'
                  ? 'OK'
                  : insights?.authenticity === 'Possibly Exaggerated'
                    ? 'FYI'
                    : insights?.authenticity === 'Suspicious Claim'
                      ? 'X'
                      : '...'}
              </span>
              <div>
                <strong>{insights?.authenticity || 'No authenticity signal yet'}</strong>
                <p>
                  Heuristic scoring combines returned severity, confidence, and metadata without altering your
                  model logic.
                </p>
              </div>
            </div>
          </article>
        </section>

        <section className="report-grid">
          <article className="card glass report-card">
            <div className="card-header">
              <div>
                <p className="section-label">Generated Claim Report</p>
                <h3>Structured summary for assessor review</h3>
              </div>
              <div className="action-row compact">
                <button className="secondary-button" type="button" onClick={handleGenerateReport} disabled={!analysis}>
                  Generate Report
                </button>
                <button className="ghost-button" type="button" onClick={handleDownloadReport} disabled={!reportText}>
                  Download
                </button>
              </div>
            </div>

            <div className="report-body">
              {reportText ? (
                <pre>{reportText}</pre>
              ) : (
                <div className="report-placeholder">
                  <strong>Report not generated yet</strong>
                  <p>Run analysis and generate a report to produce a claim-ready summary for underwriting teams.</p>
                </div>
              )}
            </div>
          </article>
        </section>
      </main>
    </div>
  );
}

export default App;
