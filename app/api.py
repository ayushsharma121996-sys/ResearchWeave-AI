"""
FastAPI app for ResearchPilot AI.

Endpoints:
  POST /papers/upload      -> upload PDFs, builds/rebuilds & persists the index
  POST /ask                -> grounded Q&A against the current index
  POST /compare             -> run full extraction + claim matrix + report
  GET  /health              -> liveness check
"""
from contextlib import asynccontextmanager
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import PAPERS_DIR, INDEX_DIR
from app.pipeline import ingest_papers, build_index, run_full_pipeline, answer_question
from app.retrieval.index import HybridIndex

# In-memory session state backed by disk persistence.
_state = {"index": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = INDEX_DIR / "index.pkl"
    if path.exists():
        try:
            index = HybridIndex()
            index.load("index")
            _state["index"] = index
            print(f"Loaded existing index from {path} ({len(index.chunks)} chunks).")
        except Exception as e:
            print(f"Warning: Failed to load index from {path}: {e}")
    yield


app = FastAPI(title="ResearchPilot AI", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class CompareRequest(BaseModel):
    topic: str = "the uploaded papers"


from fastapi.responses import HTMLResponse

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ResearchPilot AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg: #0e1117;
            --sidebar-bg: #161b22;
            --card-bg: #1f242d;
            --accent: #ff4b4b;
            --text: #fafafa;
            --text-muted: #8b949e;
            --border: #30363d;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            display: flex;
            height: 100vh;
        }
        .sidebar {
            width: 300px;
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border);
            padding: 24px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .sidebar h2 { font-size: 1.1rem; margin-top: 0; }
        .upload-area {
            background: var(--card-bg);
            border: 2px dashed var(--border);
            border-radius: 8px;
            padding: 24px 16px;
            text-align: center;
            cursor: pointer;
        }
        .upload-area:hover { border-color: var(--accent); }
        .btn {
            background-color: var(--card-bg);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 10px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
        }
        .btn-primary {
            background-color: var(--accent);
            border: none;
            color: #fff;
        }
        .main {
            flex: 1;
            padding: 40px;
            overflow-y: auto;
            max-width: 900px;
        }
        .header {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
        }
        .logo { font-size: 2.5rem; }
        .title-container h1 { margin: 0; font-size: 2rem; }
        .subtitle { color: var(--text-muted); margin-top: 4px; }
        .tabs { display: flex; gap: 24px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
        .tab { padding: 12px 0; cursor: pointer; color: var(--text-muted); font-weight: 500; border-bottom: 2px solid transparent; }
        .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            box-sizing: border-box;
            margin-bottom: 16px;
        }
        .output-box {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            line-height: 1.6;
        }
        table { border-collapse: collapse; width: 100%; margin: 16px 0; }
        th, td { border: 1px solid var(--border); padding: 8px 12px; text-align: left; }
        th { background: var(--sidebar-bg); }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>1. Upload papers</h2>
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <p><strong>Upload PDFs</strong></p>
            <p style="font-size: 0.8rem; color: var(--text-muted);">Select research papers</p>
            <input type="file" id="fileInput" multiple accept=".pdf" style="display:none" onchange="uploadFiles()">
        </div>
        <button class="btn" onclick="document.getElementById('fileInput').click()">Upload Files</button>
        <div id="status" style="font-size:0.85rem; color: var(--text-muted);"></div>
    </div>
    <div class="main">
        <div class="header">
            <div class="logo">📚</div>
            <div class="title-container">
                <h1>ResearchPilot AI</h1>
                <div class="subtitle">Upload research papers → ask grounded questions → get a structured comparison report.</div>
            </div>
        </div>
        <div class="tabs">
            <div class="tab active" onclick="switchTab('ask')">💬 Ask a question</div>
            <div class="tab" onclick="switchTab('compare')">📊 Compare papers</div>
        </div>
        <div id="tab-ask" class="tab-content active">
            <p><strong>Ask a grounded question about the uploaded papers</strong></p>
            <input type="text" id="askInput" placeholder="What is LoRA and how does it reduce memory?">
            <button class="btn btn-primary" onclick="submitAsk()">Ask</button>
            <div id="askOutput" class="output-box" style="display:none;"></div>
        </div>
        <div id="tab-compare" class="tab-content">
            <p><strong>Topic for the report</strong></p>
            <input type="text" id="compareInput" value="the uploaded papers">
            <button class="btn btn-primary" onclick="submitCompare()">Generate comparison report</button>
            <div id="compareOutput" class="output-box" style="display:none;"></div>
        </div>
    </div>
    <script>
        function switchTab(name) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            if(name==='ask'){
                document.querySelectorAll('.tab')[0].classList.add('active');
                document.getElementById('tab-ask').classList.add('active');
            } else {
                document.querySelectorAll('.tab')[1].classList.add('active');
                document.getElementById('tab-compare').classList.add('active');
            }
        }
        async function uploadFiles() {
            const input = document.getElementById('fileInput');
            if(!input.files.length) return;
            const formData = new FormData();
            for(let f of input.files) formData.append('files', f);
            document.getElementById('status').innerText = 'Indexing papers...';
            try {
                const res = await fetch('/papers/upload', {method: 'POST', body: formData});
                const data = await res.json();
                if(res.ok) {
                    document.getElementById('status').innerText = `Indexed ${data.papers_ingested.length} paper(s) (${data.total_chunks} chunks).`;
                } else {
                    document.getElementById('status').innerText = 'Upload error: ' + (data.detail || 'Failed');
                }
            } catch(e) {
                document.getElementById('status').innerText = 'Error uploading: ' + e;
            }
        }
        async function submitAsk() {
            const q = document.getElementById('askInput').value;
            if(!q) return;
            const out = document.getElementById('askOutput');
            out.style.display = 'block';
            out.innerText = 'Thinking...';
            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: q})
                });
                const data = await res.json();
                if(res.ok) {
                    out.innerHTML = marked.parse(data.answer);
                } else {
                    out.innerText = 'Error: ' + (data.detail || 'Failed');
                }
            } catch(e) {
                out.innerText = 'Error: ' + e;
            }
        }
        async function submitCompare() {
            const topic = document.getElementById('compareInput').value;
            const out = document.getElementById('compareOutput');
            out.style.display = 'block';
            out.innerText = 'Extracting structured data, cross-checking claims, generating report...';
            try {
                const res = await fetch('/compare', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({topic: topic})
                });
                const data = await res.json();
                if(res.ok) {
                    out.innerHTML = marked.parse(data.report_markdown);
                } else {
                    out.innerText = 'Error: ' + (data.detail || 'Failed');
                }
            } catch(e) {
                out.innerText = 'Error: ' + e;
            }
        }
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTML_UI


@app.get("/health")
def health():


    indexed = _state["index"] is not None
    chunks_count = len(_state["index"].chunks) if indexed else 0
    return {"status": "ok", "index_loaded": indexed, "total_chunks": chunks_count}


@app.post("/papers/upload")
async def upload_papers(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files uploaded.")

    saved_paths = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"{f.filename} is not a PDF.")
        dest = PAPERS_DIR / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(dest)

    try:
        chunks = ingest_papers(saved_paths)
        if not chunks:
            raise HTTPException(400, "No text chunks could be extracted from uploaded PDFs.")
        index = build_index(chunks)
        index.save("index")
        _state["index"] = index

        paper_ids = sorted({c.paper_id for c in chunks})
        return {
            "papers_ingested": paper_ids,
            "total_chunks": len(chunks),
        }
    except Exception as e:
        raise HTTPException(500, f"Error indexing papers: {str(e)}")


@app.post("/ask")
def ask(req: AskRequest):
    if _state["index"] is None:
        raise HTTPException(400, "No papers indexed yet. Upload papers via /papers/upload first.")
    if not req.question.strip():
        raise HTTPException(400, "Question string cannot be empty.")
    try:
        return answer_question(_state["index"], req.question)
    except Exception as e:
        raise HTTPException(500, f"Error answering question: {str(e)}")


@app.post("/compare")
def compare(req: CompareRequest):
    pdf_paths = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise HTTPException(400, "No papers found in paper repository. Upload PDFs first.")

    try:
        result = run_full_pipeline(pdf_paths, topic=req.topic)
        result["index"].save("index")
        _state["index"] = result["index"]

        return {
            "report_markdown": result["report"],
            "num_papers": len({e.paper_id for e in result["extractions"]}),
            "num_claims_analyzed": len(result["claim_matrix"]),
        }
    except Exception as e:
        raise HTTPException(500, f"Error generating comparison report: {str(e)}")

