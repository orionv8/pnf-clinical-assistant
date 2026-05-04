# api.py — FastAPI bridge for PNF Clinical Assistant
#
# Endpoints
#   GET  /              → serves index.html (chatbot frontend) or holding page
#   GET  /health        → liveness probe
#   POST /api/pnf/ask   → drug search with PNF + Gemini AI integration
#
# Features:
# - PNF drug index search (exact → partial → content match)
# - Gemma/Vertex AI synthesis (drug interaction queries: "X and Y", "X vs Y")
# - HTML response formatting with sections + citations
# - AMS Restricted Antimicrobial alerts
# - Sources array with PNF freshness metadata
# - Empty-query validation
# - BOM stripping
#
# Required environment variables (Cloud Run / .env):
#   PROJECT_ID            — GCP project ID for Vertex AI
#   LOCATION              — Vertex AI region (e.g. asia-southeast1)
#   MODEL_NAME            — Gemma model name (e.g. gemma-2-9b-it, gemini-1.5-flash)

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
from brave_resolver import brave_resolve_generic
from ai_resolver import ai_resolve_generic
import os
import re
import hashlib
import secrets
import time

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Optional integrations: Vertex AI (Gemini)
# Wrapped in try/except so the API still serves PNF queries if env vars
# or packages are missing.
# ---------------------------------------------------------------------------

# Vertex AI / Gemma (heavier, requires GCP auth)
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
except Exception as _e:
    # Gemma not available — interaction queries will return a graceful fallback
    _GEMMA_MODEL = None

# ---------------------------------------------------------------------------
BRAVE_KEY = os.getenv("BRAVE_SEARCH")

# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PNF Clinical Assistant API",
    description="REST bridge for the Philippine National Formulary with Gemini AI integration.",
    version="1.1.0",
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
# PNF index
# ---------------------------------------------------------------------------

pnf_data = []

index_path = os.path.join(BASE_DIR, "data", "pnf_index.json")
if os.path.exists(index_path):
    with open(index_path, "r", encoding="utf-8") as _f:
        pnf_data = json.load(_f)

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

class AuthRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    token: str
    email: str
    message: str

# ---------------------------------------------------------------------------
# In-memory auth store
# NOTE: resets on container restart — replace with a real DB for production.
# ---------------------------------------------------------------------------
_users: dict = {}   # email -> { password_hash, created_at }
_tokens: dict = {}  # token -> { email, expires }

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _create_token(email: str) -> str:
    token = secrets.token_urlsafe(32)
    _tokens[token] = {"email": email, "expires": time.time() + 365 * 86400}
    return token

def _resolve_token(authorization: Optional[str]) -> Optional[dict]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:].strip()
    entry = _tokens.get(token)
    if not entry:
        return None
    if time.time() > entry["expires"]:
        del _tokens[token]
        return None
    email = entry["email"]
    return {"email": email} if email in _users else None

# ---------------------------------------------------------------------------
# PNF helpers
# ---------------------------------------------------------------------------

def _clean_text(raw):
    # Strip BOM, publisher boilerplate, ATC codes, page markers
    cleaned = raw.lstrip("\ufeff")
    return re.sub(
        r"April.*?\n"
        r"|https://.*?pnf\.doh\.gov\.ph\n+"
        r"|ATC CODE\n+.*?\n+"
        r"|Page \d of \d",
        "", cleaned,
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

def _markdown_to_html(text):
    """Convert basic Gemini markdown formatting to HTML."""
    # Bold: **text** → <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic: *text* → <em>text</em> (skip if already inside <strong>)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    # Bullet points: lines starting with "* " or "- "
    text = re.sub(r'^\*\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^-\s+', '• ', text, flags=re.MULTILINE)
    return text

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
            parts.append(f"<p>{_markdown_to_html(stripped)}</p>")
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

def _build_ai_notice():
    return (
        '<p class="ai-notice" style="'
        'background:#e7f3ff;border-left:4px solid #2196f3;'
        'padding:0.6em 0.8em;border-radius:4px;margin-bottom:0.8em;'
        '">'
        '<strong>&#9432; AI-Synthesized Summary</strong> \u2014 '
        'This is an AI-generated reference. Cross-check with the PNF and '
        'clinical references before prescribing.'
        '</p>'
    )




# ---------------------------------------------------------------------------
# Gemma / Vertex AI: drug interaction synthesis
# ---------------------------------------------------------------------------

def synthesize_interaction(drugs):
    """
    Use Gemma (Vertex AI) to generate a clinical interaction summary for
    the given list of drugs. Returns the AI-generated text, or raises if
    Gemma is unavailable.
    """
    if _GEMMA_MODEL is None:
        raise RuntimeError(
            "Gemma not configured (set PROJECT_ID, LOCATION, MODEL_NAME env vars)."
        )
    prompt = (
        "Provide a concise clinical drug interaction summary for the following "
        "medications: "
        + ", ".join(drugs)
        + ". Cover mechanism, severity, and clinical management in plain text. "
        "Do not include disclaimers — they are added by the UI."
    )
    return _GEMMA_MODEL.generate_content(prompt).text

def is_interaction_query(query):
    """Detect drug interaction queries like 'X and Y' or 'X vs Y'."""
    q = query.lower()
    return " and " in q or " vs " in q or " versus " in q

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# Holding page shown when full index.html hasn't been deployed yet
HOLDING_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PNF Clinical Assistant</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #08090a; color: #f7f8f8; display: flex;
           align-items: center; justify-content: center;
           min-height: 100vh; margin: 0; }
    .card { background: #0f1011; border: 1px solid rgba(255,255,255,.08);
            border-radius: 14px; padding: 40px; max-width: 480px; text-align: center; }
    h1 { font-size: 28px; margin: 0 0 12px; color: #5e6ad2; }
    p { color: #8a8f98; line-height: 1.6; margin: 0 0 20px; }
    .status { display: inline-block; background: #1a3a1a; color: #4ade80;
              padding: 6px 16px; border-radius: 999px; font-size: 14px; }
    .links { margin-top: 24px; }
    a { color: #5e6ad2; text-decoration: none; margin: 0 10px; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="card">
    <h1>PNF Clinical Assistant</h1>
    <p>The API is running with {entries} drug entries loaded.<br>
    The chatbot frontend (index.html) is being deployed.</p>
    <span class="status">&#10003; API Online</span>
    <div class="links"><a href="/health">Health</a> <a href="/docs">API Docs</a></div>
  </div>
</body>
</html>"""

@app.get("/")
async def serve_frontend():
    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path) and os.path.getsize(html_path) > 10000:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    return HTMLResponse(
        content=HOLDING_PAGE.replace("{entries}", str(len(pnf_data)))
    )

@app.post("/api/auth/register", response_model=AuthResponse)
async def register(req: AuthRequest):
    """Create a new account. Returns a Bearer token valid for 1 year."""
    email = req.email.strip().lower()
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=422, detail="A valid email address is required.")
    if len(req.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")
    if email in _users:
        raise HTTPException(status_code=409, detail="Email already registered. Please sign in instead.")
    _users[email] = {"password_hash": _hash_password(req.password), "created_at": time.time()}
    token = _create_token(email)
    return AuthResponse(token=token, email=email, message="Account created successfully.")

@app.post("/api/auth/login", response_model=AuthResponse)
async def login(req: AuthRequest):
    """Sign in to an existing account. Returns a fresh Bearer token."""
    email = req.email.strip().lower()
    user = _users.get(email)
    if not user:
        raise HTTPException(status_code=401, detail="Email not found. Please create an account.")
    if user["password_hash"] != _hash_password(req.password):
        raise HTTPException(status_code=401, detail="Incorrect password.")
    token = _create_token(email)
    return AuthResponse(token=token, email=email, message="Signed in successfully.")

@app.get("/api/auth/me")
async def get_me(authorization: Optional[str] = Header(None)):
    """Returns the current user if the Bearer token is valid."""
    user = _resolve_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "ok",
        "entries_loaded": len(pnf_data),
        "gemma_available": _GEMMA_MODEL is not None,
        "brave_available": bool(BRAVE_KEY) and _HAS_REQUESTS,
    })

@app.post("/api/pnf/ask", response_model=AskResponse)
async def ask(request: AskRequest, authorization: Optional[str] = Header(None)):
    # Authenticated users bypass rate-limiting (the client enforces the
    # 7-try gate; the token is also validated here for server-side trust).
    user = _resolve_token(authorization)  # None for anonymous users
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="'question' must not be empty.")

    # ----------------------------------------------------------------
    # Path A: Drug interaction query → use Gemma
    # ----------------------------------------------------------------
    if is_interaction_query(question):
        drugs = [d.strip() for d in re.split(r" and | vs | versus ", question.lower()) if d.strip()]
        try:
            ai_text = synthesize_interaction(drugs)
            ai_html = _format_text_as_html(ai_text)
            body_html = "\n".join([_build_ai_notice(), ai_html])
            return AskResponse(
                body=body_html,
                sources=[
                    SourceItem(
                        num=1,
                        title="AI-Synthesized Summary",
                        section=f"Drug Interaction: {' + '.join(drugs)}",
                        snippet="Generated by Gemma (Vertex AI) for clinical reference. Always cross-check with PNF and authoritative sources.",
                        lastUpdated="Live",
                    )
                ],
            )
        except Exception as e:
            err_html = (
                f"<p>Unable to synthesize interaction summary: {str(e)[:120]}</p>"
                "<p>Try searching each drug individually in the PNF library.</p>"
            )
            return AskResponse(body=err_html, sources=[])

    # ----------------------------------------------------------------
    # Path B: Single-drug PNF lookup with AI brand resolver
    # ----------------------------------------------------------------
    match = _search_index(question)
    used_resolver = "none" # "gemma", "brave"
    resolved_name = question

    # Try Brave FIRST to resolve brand -> generic
    if match is None:
        generic_via_brave = brave_resolve_generic(question, pnf_data)
        if generic_via_brave:
            match = _search_index(generic_via_brave)
            if match is not None:
                used_resolver = "brave"
                resolved_name = generic_via_brave

    # Try Gemma ONLY as a last resort
    if match is None:
        generic_via_gemma = ai_resolve_generic(question, _GEMMA_MODEL)
        if generic_via_gemma:
            match = _search_index(generic_via_gemma)
            if match is not None:
                used_resolver = "gemma"
                resolved_name = generic_via_gemma


    if match is None:
        not_found_html = (
            f"<p>No information found for <strong>{question}</strong> in the "
            "Philippine National Formulary index. "
            "Please verify the drug name spelling or try a generic name.</p>"
        )
        return AskResponse(body=not_found_html, sources=[])

    # ----------------------------------------------------------------
    # Format successful PNF match (with optional AI provenance)
    # ----------------------------------------------------------------
    raw_text = match.get("text", "")
    drug_name = match.get("drug", question)
    clean_text = _clean_text(raw_text)

    citation_link = _build_citation_link(1, drug_name)
    ams_alert = _build_ams_alert(drug_name)
    monograph_html = _format_text_as_html(clean_text)

    body_parts = []

    # If a resolver was used, show a note explaining the brand->generic mapping
    if used_resolver != "none":
        resolver_name = "Gemma AI" if used_resolver == "gemma" else "Brave Search"
        body_parts.append(
            f'<p class="resolver-notice" style="'
            f'background:#e6ffed;border-left:4px solid #4ade80;'
            f'padding:0.6em 0.8em;border-radius:4px;margin-bottom:0.8em;font-size:0.9em;'
            f'">'
            f'<strong>Brand &rarr; Generic:</strong> "{question}" was resolved to '
            f'<em>{drug_name}</em> via {resolver_name}. Showing PNF data for the generic.'
            f'</p>'
        )

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
