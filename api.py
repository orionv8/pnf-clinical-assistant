# api.py — FastAPI bridge for PNF Clinical Assistant (OPTIMIZED)

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import re
import hashlib
import secrets
import time

from functools import lru_cache
from rapidfuzz import process, fuzz

from brave_resolver import brave_resolve_generic
from ai_resolver import ai_resolve_generic

# ---------------------------------------------------------------------------
# Optional integrations: Vertex AI
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_GEMMA_MODEL = None
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    project = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")
    model_name = os.getenv("MODEL_NAME")
    if project and location and model_name:
        vertexai.init(project=project, location=location)
        _GEMMA_MODEL = GenerativeModel(model_name)
except Exception:
    _GEMMA_MODEL = None

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="PNF Clinical Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data + Optimized Indexes
# ---------------------------------------------------------------------------
pnf_data = []

drug_index = {}
prefix_index = {}
content_index = {}
drug_names = []

def _clean_text(raw):
    cleaned = raw.lstrip("\ufeff")
    return re.sub(
        r"April.*?\n|https://.*?pnf\.doh\.gov\.ph\n+|ATC CODE\n+.*?\n+|Page \d of \d",
        "",
        cleaned,
    )

index_path = os.path.join(BASE_DIR, "data", "pnf_index.json")

if os.path.exists(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
        pnf_data = json.load(f)

    for entry in pnf_data:
        drug = entry.get("drug", "").lower().strip()
        text = entry.get("text", "")

        if not drug:
            continue

        clean = _clean_text(text)
        entry["clean_text"] = clean

        drug_index[drug] = entry
        drug_names.append(drug)

        for i in range(1, min(len(drug), 10) + 1):
            prefix = drug[:i]
            prefix_index.setdefault(prefix, []).append(entry)

        words = set(re.findall(r"\b\w+\b", clean.lower()))
        for word in words:
            if len(word) > 2:
                content_index.setdefault(word, []).append(entry)

# ---------------------------------------------------------------------------
# Models
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
# Auth (same as before)
# ---------------------------------------------------------------------------
_users = {}
_tokens = {}

def _hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def _create_token(email):
    token = secrets.token_urlsafe(32)
    _tokens[token] = {"email": email, "expires": time.time() + 365 * 86400}
    return token

def _resolve_token(auth):
    if not auth or not auth.startswith("Bearer "):
        return None
    token = auth[7:].strip()
    entry = _tokens.get(token)
    if not entry or time.time() > entry["expires"]:
        return None
    return {"email": entry["email"]}

# ---------------------------------------------------------------------------
# 🔥 OPTIMIZED SEARCH
# ---------------------------------------------------------------------------
@lru_cache(maxsize=500)
def _search_index(query: str):
    q = query.lower().strip()
    if not q:
        return None

    # Exact
    if q in drug_index:
        return drug_index[q]

    # Prefix
    if q in prefix_index:
        return prefix_index[q][0]

    # Fuzzy
    if drug_names:
        match, score, _ = process.extractOne(q, drug_names, scorer=fuzz.WRatio)
        if score >= 85:
            return drug_index[match]

    # Content
    words = re.findall(r"\b\w+\b", q)
    candidates = []

    for word in words:
        if word in content_index:
            candidates.extend(content_index[word])

    if candidates:
        def score_entry(e):
            score = 0
            drug = e["drug"].lower()
            text = e["clean_text"].lower()
            if q in drug:
                score += 50
            if q in text:
                score += 10
            return score

        candidates = sorted(set(candidates), key=score_entry, reverse=True)
        return candidates[0]

    return None

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _format_text_as_html(text):
    return "\n".join(f"<p>{line}</p>" for line in text.splitlines() if line.strip())

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "entries": len(pnf_data),
        "optimized_search": True
    }

@app.post("/api/pnf/ask", response_model=AskResponse)
async def ask(req: AskRequest, authorization: Optional[str] = Header(None)):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=422, detail="Empty query")

    match = _search_index(question)

    if not match:
        return AskResponse(
            body=f"<p>No results for <strong>{question}</strong></p>",
            sources=[]
        )

    drug_name = match.get("drug", question)
    clean_text = match.get("clean_text", "")

    snippet = clean_text[:200] + "..."

    return AskResponse(
        body=_format_text_as_html(clean_text),
        sources=[
            SourceItem(
                num=1,
                title="Philippine National Formulary",
                section=f"{drug_name}",
                snippet=snippet,
                lastUpdated="Apr 2026"
            )
        ]
    )

# ---------------------------------------------------------------------------
# Dev
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8501, reload=True)
