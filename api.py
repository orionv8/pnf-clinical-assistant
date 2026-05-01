# api.py — FastAPI bridge for PNF Clinical Assistant
#
# Wraps the same search logic used by the Streamlit app (app.py) and
# exposes it via a REST endpoint so the pure-HTML frontend can call it
# without any Streamlit dependency at runtime.
#
# Endpoints
#   GET  /          → serves index.html (frontend)
#   GET  /health    → liveness probe
#   POST /api/pnf/ask → main search endpoint

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import re

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PNF Clinical Assistant API",
    description="REST bridge for the Philippine National Formulary drug monograph index.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMS_RESTRICTED = [
    "cefepime", "ertapenem", "meropenem", "vancomycin",
    "amphotericin b", "voriconazole", "colistin", "micafungin",
    "aztreonam", "linezolid", "imipenem", "tigecycline",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# PNF index — loaded once at startup
# ---------------------------------------------------------------------------

pnf_data = []

index_path = os.path.join(BASE_DIR, "data", "pnf_index.json")
if os.path.exists(index_path):
    with open(index_path, "r", encoding="utf-8") as _f:
        pnf_data = json.load(_f)
else:
    import warnings
    warnings.warn(
        f"PNF index not found at {index_path}. "
        "Place data/pnf_index.json next to api.py before serving queries.",
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    num: int
    title: str
    section: str
    snippet: str
    lastUpdated: str

class AskResponse(BaseModel):
    body: str
    sources: List[SourceItem]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_text(raw):
    return re.sub(
        r"April.*?\n"
        r"|https://.*?pnf\.doh\.gov\.ph\n+"
        r"|ATC CODE\n+.*?\n+"
        r"|Page \d of \d",
        "", raw,
    )

def _search_index(query):
    q = query.lower().strip()
    exact, partial, content = [], [], []
    for entry in pnf_data:
        drug_lower = entry.get("drug", "").lower().strip()
        text_lower = entry.get("text", "").lower()
        if q == drug_lower:
            exact.append(entry)
        elif q in drug_lower:
            partial.append(entry)
        elif q in text_lower:
            content.append(entry)
    for bucket in (exact, partial, content):
        if bucket:
            return sorted(bucket, key=lambda e: len(e.get("text", "")))[0]
    return None

def _format_text_as_html(text):
    lines = text.splitlines()
    parts = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        core = stripped.rstrip(":.")
        if len(core) >= 3 and core == core.upper() and core.replace(" ", "").isalpha():
            parts.append(f"<p><strong>{stripped}</strong></p>")
        else:
            parts.append(f"<p>{stripped}</p>")
    return "\n".join(parts)

def _build_ams_alert(drug_name):
    if drug_name.lower().strip() in AMS_RESTRICTED:
        return (
            '<p class="ams-alert" style="'
            'background:#fff3cd;border-left:4px solid #ffc107;'
            'padding:0.6em 0.8em;border-radius:4px;margin-bottom:0.8em;'
            '">' +
            '<strong>&#9888; AMS Restricted Antimicrobial</strong> \u2014 ' +
            f'<em>{drug_name}</em> requires AMS clearance and documented indication.' +
            '</p>'
        )
    return ""

def _build_citation_link(num, drug_name):
    return (
        f'<a class="citation" href="#source-{num}" '
        f'title="Philippine National Formulary \u2014 {drug_name}">'
        f'[{num}]</a>'
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_frontend():
    html_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(html_path, media_type="text/html")

@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok", "entries_loaded": len(pnf_data)})

@app.post("/api/pnf/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="'question' must not be empty.")

    match = _search_index(question)

    if match is None:
        not_found_html = (
            f"<p>No information found for <strong>{question}</strong> in the "
            "Philippine National Formulary index. "
            "Please verify the drug name spelling or try a generic name.</p>"
        )
        return AskResponse(body=not_found_html, sources=[])

    raw_text = match.get("text", "")
    drug_name = match.get("drug", question)
    clean_text = _clean_text(raw_text)

    citation_link = _build_citation_link(1, drug_name)
    ams_alert = _build_ams_alert(drug_name)
    monograph_html = _format_text_as_html(clean_text)

    body_parts = []
    if ams_alert:
        body_parts.append(ams_alert)
    body_parts.append(monograph_html)
    body_parts.append(f"<p>{citation_link}</p>")
    body_html = "\n".join(body_parts)

    snippet = clean_text[:200].strip()
    if len(clean_text) > 200 and not clean_text[200].isspace():
        last_space = snippet.rfind(" ")
        if last_space > 0:
            snippet = snippet[:last_space]
    snippet = snippet + "\u2026" if len(clean_text) > 200 else snippet

    sources = [
        SourceItem(
            num=1,
            title="Philippine National Formulary",
            section=f"Drug Monograph \u2014 {drug_name}",
            snippet=snippet,
            lastUpdated="Apr 2026",
        )
    ]

    return AskResponse(body=body_html, sources=sources)

# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8501, reload=True)
