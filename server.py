"""
server.py - Interactive Web Dashboard & REST API for Cortex Personal Brain.

Features:
- REST API endpoints for live semantic querying, timeline shifts, and confidence scoring.
- Interactive HTML5/JavaScript dashboard with interactive timeline, Louvain community explorer,
  and verbatim quote provenance inspector.
- Runs via: `python server.py` or `uvicorn server:app --port 8000`
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from config import settings
from pipeline import CortexPersonalBrain
from graph_engine import HierarchicalTemporalGraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cortex.server")

app = FastAPI(
    title="Cortex Personal Brain - Dr. Michael Snyder",
    description="Scientific Historian AI with Anti-Flattening GraphRAG and Mathematical Confidence",
    version="2.0.0"
)

# Global Brain Instance
brain: Optional[CortexPersonalBrain] = None


@app.on_event("startup")
def startup_event():
    global brain
    logger.info("Initializing Cortex Personal Brain on startup...")
    brain = CortexPersonalBrain()
    logger.info("Cortex Personal Brain initialized and ready for queries.")


class QueryRequest(BaseModel):
    question: str


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Returns database, vector index, and knowledge graph metrics."""
    global brain
    if brain is None:
        brain = CortexPersonalBrain()

    raw_dir = settings.raw_data_dir
    raw_files = [f for f in raw_dir.iterdir() if f.is_file() and not f.name.endswith(".meta.json") and f.name != "scraper_manifest.json"]
    
    total_chunks = len(brain.chunk_database)
    node_count = brain.hierarchical_graph.graph.number_of_nodes()
    edge_count = brain.hierarchical_graph.graph.number_of_edges()
    community_count = len(brain.hierarchical_graph.community_summaries)

    return {
        "verified_documents_on_disk": len(raw_files),
        "active_chunks": total_chunks,
        "knowledge_graph_nodes": node_count,
        "knowledge_graph_edges": edge_count,
        "hierarchical_communities": community_count,
        "sample_communities": brain.hierarchical_graph.community_summaries[:10]
    }


@app.get("/api/communities")
def get_communities() -> Dict[str, Any]:
    """Returns all pre-computed Louvain research communities and macro summaries."""
    global brain
    if brain is None:
        brain = CortexPersonalBrain()
    return {
        "count": len(brain.hierarchical_graph.community_summaries),
        "communities": brain.hierarchical_graph.community_summaries
    }


@app.post("/api/query")
def execute_query(req: QueryRequest) -> Dict[str, Any]:
    """Executes end-to-end anti-flattening query with exact citation verification."""
    global brain
    if brain is None:
        brain = CortexPersonalBrain()

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = brain.query(req.question.strip())
    return result


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the complete interactive web dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cortex Personal Brain | Dr. Michael Snyder</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --accent-emerald: #10b981;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding-bottom: 50px;
        }
        .navbar {
            background-color: #090d16;
            border-bottom: 1px solid var(--border-color);
        }
        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .badge-cyan { background-color: rgba(6, 182, 212, 0.2); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); }
        .badge-emerald { background-color: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); border: 1px solid var(--accent-emerald); }
        .timeline-item {
            position: relative;
            padding-left: 28px;
            margin-bottom: 18px;
            border-left: 2px solid var(--accent-cyan);
        }
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -6px;
            top: 2px;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: var(--accent-cyan);
        }
        .quote-box {
            background-color: rgba(15, 23, 42, 0.8);
            border-left: 3px solid var(--accent-emerald);
            padding: 10px 14px;
            margin-top: 8px;
            font-size: 0.92rem;
            border-radius: 4px;
        }
        .confidence-meter {
            height: 8px;
            border-radius: 4px;
            background-color: #334155;
            overflow: hidden;
        }
        .confidence-bar {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #10b981);
            transition: width 0.6s ease;
        }
        .stat-val { font-size: 1.8rem; font-weight: 700; color: var(--accent-cyan); }
        .stat-label { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
        .btn-query { background: linear-gradient(135deg, #06b6d4, #3b82f6); color: white; border: none; font-weight: 600; }
        .btn-query:hover { opacity: 0.9; color: white; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark py-3">
        <div class="container">
            <span class="navbar-brand mb-0 h1 d-flex align-items-center">
                <span style="font-size: 1.5rem; margin-right: 10px;">🧠</span>
                <div>
                    <strong>CORTEX PERSONAL BRAIN</strong>
                    <span class="badge badge-cyan ms-2" style="font-size: 0.7rem;">v2.0 Hierarchical GraphRAG</span>
                </div>
            </span>
            <span class="text-muted small">Stanford Systems Biology & Genomics (1981–2026)</span>
        </div>
    </nav>

    <div class="container mt-4">
        <!-- System Metrics Row -->
        <div class="row g-3 mb-4" id="stats-row">
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="stat-val" id="doc-count">1,764</div>
                    <div class="stat-label">Verified Documents</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="stat-val" id="chunk-count">17,372</div>
                    <div class="stat-label">Deduplicated Chunks</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="stat-val" id="graph-count">68,686</div>
                    <div class="stat-label">Epistemic Graph Edges</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="stat-val" id="community-count">322</div>
                    <div class="stat-label">Louvain Communities</div>
                </div>
            </div>
        </div>

        <!-- Query Input Card -->
        <div class="card p-4">
            <h5 class="mb-3">Ask Michael Snyder's Personal Brain</h5>
            <div class="input-group mb-3">
                <input type="text" id="query-input" class="form-control form-control-lg bg-dark text-white border-secondary" 
                       placeholder="e.g. Why did I move from Drosophila to yeast? Or: How did wearables change baseline health?" 
                       value="Why did I move from Drosophila to yeast?">
                <button class="btn btn-query px-4" id="submit-btn" onclick="runQuery()">Ask Question</button>
            </div>
            <div class="d-flex flex-wrap gap-2 small">
                <span class="text-muted">Preset Landmark Questions:</span>
                <a href="#" class="text-info text-decoration-none" onclick="setQuery('Why did I move from Drosophila to yeast?')">Drosophila to Yeast</a> •
                <a href="#" class="text-info text-decoration-none" onclick="setQuery('How did wearable biosensors and continuous monitoring reshape my definition of baseline health?')">Wearables & Baseline Health</a> •
                <a href="#" class="text-info text-decoration-none" onclick="setQuery('What were the biggest obstacles when introducing DNA microarrays and ChIP-chip to the community?')">Microarrays & ChIP-chip</a>
            </div>
        </div>

        <!-- Loading Indicator -->
        <div id="loading" class="text-center my-4 d-none">
            <div class="spinner-border text-info" role="status"></div>
            <p class="text-muted mt-2">Traversing Hierarchical Graph, Stratifying Chronological Chunks & Verifying Substring Quotes...</p>
        </div>

        <!-- Results Container -->
        <div id="results" class="d-none">
            <!-- Narrative Card -->
            <div class="card p-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h5 class="mb-0 text-info">Forensic Synthesis & Reasoning Evolution</h5>
                    <span id="confidence-badge" class="badge badge-emerald py-2 px-3">Confidence: 75.0%</span>
                </div>
                <div id="narrative-text" style="line-height: 1.7; font-size: 1.05rem; white-space: pre-wrap;"></div>
            </div>

            <!-- Timeline Shifts & Evidence -->
            <div class="row">
                <div class="col-md-7">
                    <div class="card p-4 h-100">
                        <h5 class="mb-3 text-warning">Historical Timeline & Epistemic Shifts</h5>
                        <div id="timeline-container"></div>
                    </div>
                </div>
                <div class="col-md-5">
                    <div class="card p-4 h-100">
                        <h5 class="mb-3 text-success">Byte-for-Byte Verified Citations</h5>
                        <div id="citations-container"></div>
                        <hr class="border-secondary my-3">
                        <h6 class="text-muted small">MATHEMATICAL CONFIDENCE BREAKDOWN</h6>
                        <div class="mb-2">
                            <div class="d-flex justify-content-between small mb-1">
                                <span>Total Confidence</span>
                                <strong id="conf-total">75%</strong>
                            </div>
                            <div class="confidence-meter">
                                <div class="confidence-bar" id="conf-bar" style="width: 75%;"></div>
                            </div>
                        </div>
                        <div class="small text-muted" id="conf-rationale"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('doc-count').innerText = data.verified_documents_on_disk.toLocaleString();
                document.getElementById('chunk-count').innerText = data.active_chunks.toLocaleString();
                document.getElementById('graph-count').innerText = data.knowledge_graph_edges.toLocaleString();
                document.getElementById('community-count').innerText = data.hierarchical_communities.toLocaleString();
            } catch(e) { console.error('Status fetch failed:', e); }
        }

        function setQuery(q) {
            document.getElementById('query-input').value = q;
            runQuery();
        }

        async function runQuery() {
            const question = document.getElementById('query-input').value.trim();
            if (!question) return;

            document.getElementById('loading').classList.remove('d-none');
            document.getElementById('results').classList.add('d-none');
            document.getElementById('submit-btn').disabled = true;

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                });
                const data = await res.json();

                // Populate Narrative
                document.getElementById('narrative-text').innerText = data.narrative;
                
                // Populate Confidence
                const conf = data.confidence;
                const score = conf.final_score.toFixed(1);
                document.getElementById('confidence-badge').innerText = `Confidence: ${score}%`;
                document.getElementById('conf-total').innerText = `${score}%`;
                document.getElementById('conf-bar').style.width = `${score}%`;
                document.getElementById('conf-rationale').innerText = conf.rationale;

                // Populate Timeline Shifts
                const tlContainer = document.getElementById('timeline-container');
                tlContainer.innerHTML = '';
                (data.timeline_shifts || []).forEach(shift => {
                    const item = document.createElement('div');
                    item.className = 'timeline-item';
                    item.innerHTML = `<strong>Year ${shift.year}</strong>: ${shift.belief}`;
                    tlContainer.appendChild(item);
                });

                // Populate Citations
                const citeContainer = document.getElementById('citations-container');
                citeContainer.innerHTML = '';
                (data.citations || []).forEach(cite => {
                    const qBox = document.createElement('div');
                    qBox.className = 'quote-box';
                    qBox.innerHTML = `<em>"${cite.exact_quote}"</em><div class="text-muted small mt-1">Chunk ID: ${cite.chunk_id.substring(0, 8)}...</div>`;
                    citeContainer.appendChild(qBox);
                });

                document.getElementById('results').classList.remove('d-none');
            } catch(e) {
                alert('Query failed: ' + e);
            } finally {
                document.getElementById('loading').classList.add('d-none');
                document.getElementById('submit-btn').disabled = false;
            }
        }

        fetchStatus();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


def main():
    """Runs the Cortex Personal Brain FastAPI Server."""
    print("=" * 80)
    print("=== STARTING CORTEX PERSONAL BRAIN INTERACTIVE WEB SERVER ===")
    print("=== Open your browser at: http://127.0.0.1:8000           ===")
    print("=" * 80)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
