# api.py — FastAPI bridge for PNF Clinical Assistant
#
# Endpoints
#   GET  /              → serves index.html (chatbot frontend) or holding page
#   GET  /health        → liveness probe
#   POST /api/pnf/ask   → drug search with PNF + Brave + Gemma integration
#
# Features:
# - PNF drug index search (exact → partial → content match)
# - Brave Search fallback (brand name → generic name translation)
# - Gemma/Vertex AI synthesis (drug interaction queries: "X and Y", "X vs Y")
# - HTML response formatting with sections + citations
# - AMS Restricted Antimicrobial alerts
# - Sources array with PNF freshness metadata
# - Empty-query validation
# - BOM stripping
# - Auth: /api/auth/register, /api/auth/login, /api/auth/me
#
# Required environment variables (Cloud Run / .env):
#   PROJECT_ID            — GCP project ID for Vertex AI
#   LOCATION              — Vertex AI region (e.g. asia-southeast1)
#   MODEL_NAME            — Gemini model name
#   BRAVE_SEARCH_API_KEY  — Brave Search API subscription token

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

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

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
    _GEMMA_MODEL = None

BRAVE_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

app = FastAPI(
    title="PNF Clinical Assistant API",
    description="REST bridge for the Philippine National Formulary with Brave + Gemini integration.",
    version="1.2.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

AMS_RESTRICTED = ["cefepime","ertapenem","meropenem","vancomycin","amphotericin b","voriconazole","colistin","micafungin","aztreonam","linezolid","imipenem","tigecycline"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pnf_data = []
index_path = os.path.join(BASE_DIR, "data", "pnf_index.json")
if os.path.exists(index_path):
    with open(index_path, "r", encoding="utf-8") as _f:
        pnf_data = json.load(_f)

# ------------------------------------------------ Models
class AskRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    num: int; title: str; section: str; snippet: str; lastUpdated: str

class AskResponse(BaseModel):
    body: str; sources: List[SourceItem]

class AuthRequest(BaseModel):
    email: str; password: str

class AuthResponse(BaseModel):
    token: str; email: str; message: str

# ------------------------------------------------ Auth store (in-memory)
_users: dict = {}
_tokens: dict = {}

def _hash_pw( pw): return hashlib.sha256(pw.encode()).hexdigest()
def _make_token(email):
    t = secrets.token_urlsafe(32)
    _tokens[t] = {"email": email, "expires": time.time() + 365*86400}
    return t
def _resolve_token(authhdr):
    if not authhdr or not authhdr.startswith("Bearer "): return None
    t = authhdr[7:].strip()
    entry = _tokens.get(t)
    if not entry: return None
    if time.time() > entry["expires"]: del _tokens[t]; return None
    email = entry["email"]
    return {"email": email} if email in _users else None

# ------------------------------------------------ Helpers
def _clean_text(raw):
    return re.sub(r"April.*?\n|https://.*?pnf\.doh\.gov\.ph\n+|ATC CODE\n+.*?\n+|Page \d of \d",
        "", raw.lstrip("\ufeff"))

def _search_index(q):
    q = q.lower().strip()
    exact, partial, cont = [], [], []
    for e in pnf_data:
        dl = e.get("drug","").lower().strip()
        if q == dl: exact.append(e)
        elif q in dl: partial.append(e)
        elif q in e.get("text","").lower(): cont.append(e)
    for b in (exact, partial, cont):
        if b: return sorted(b, key=lambda x: len(x.get("text","")))[0]
    return None

def _md_to_html(t):
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', t)
    t = re.sub(r'^[*-]\s+', '✢ ', t, flags=re.MULTILINE)
    return t

def _format_html(text):
    parts = []
    for line in text.splitlines():
        s = line.strip()
        if not s: continue
        c = s.rstrip(":.")
        if len(c) >= 3 and c == c.upper() and c.replace(" ","").isalpha():
            parts.append(f"<p><strong>{s}</strong></p>")
        else:
            parts.append(f"<p>{_md_to_html(s)}</p>")
    return "\n".join(parts)

def _ams_alert(n):
    if n.lower().strip() in AMS_RESTRICTED:
        return '<p style="background:#fff3cd;border-left:4px solid #ffc107;padding:0.6em 0.8em;border-radius:4px;margin-bottom:0.8em"><strong>&#9888; AMSRestricted Antimicrobial</strong> \u2014 <em>' + n + '</em> requires AMS clearance and documented indication.</p>'
    return ""

def _citation(num, n):
    return f'<a class="citation" href="#source-{num}" title="PNF \u2014 {n}">[{num}]</a>'

def _ai_notice():
    return '<p style="background:#e7f3ff;border-left:4px solid #2196f3;padding:0.6em 0.8em;border-radius:4px;margin-bottom:0.8em"><strong>&#9432; AI-Synthesized Summary</strong> \u2014 AI-generated reference. Cross-check with PNF and clinical references before prescribing.</p>'

# ------------------------------------------------ Brave Search
def brave_search_generic(brand):
    if not BRAVE_KEY or not _HAS_REQUESTS: return None
    try:
        res = requests.get(
            f"https://api.search.brave.com/res/v1/web/search?q={brand}+Philippines+drug+generic",
            headers={"X-Subscription-Token": BRAVE_KEY}, timeout=5
        )
        if res.status_code == 200:
            results = res.json().get("web",{}).get("results",[])
            for r (in results[:3]):
                for src in (r.get("title",""), r.get("description","")):
                    g = re.split(r'\|\-\(\,\:', src)[0].strip().lower()
                    if 2 < len(g) < 40: return g
    except Exception: pass
    return None

# ------------------------------------------------ Gemini
def synthesize(drugs):
    if _GEMMA_MODEL is None: raise RuntimeError("Gemini not configured.")
    return _GEMMA_MODEL.generate_content(
        f"Concise clinical drug interaction summary for: {', '.join(drugs)}."
        " Cover mechanism, severity, management. No disclaimers."
    ).text

def is_interaction(q):
    q = q.lower()
    return " and " in q or " vs " in q or " versus " in q

HOLDING_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>PNF Clinical Assistant</title><style>body{font-family:sans-serif;background:#08090a;color:#f7f8f8;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}.card{background:#0f1011;border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:40px;max-width:480px;text-align:center}h1{font-size:28px;margin:0 0 12px;color:#5e6ad2}p{color:#8a8f98;line-height:1.6;margin:0 0 20px}.status{display:inline-block;background:#1a3a1a;color:#4ade80;padding:6px 16px;border-radius:999px;font-size:14px}.links{margin-top:24px}a{color:#5e6ad2;text-decoration:none;margin:0 10px}</style></head><body><div class="card"><h1>PNF Clinical Assistant</h1><p >API running with {entries} entries.<br>Deploying frontend...</p><span class="status">&#10003; API Online</span><div class="links"><a href="/health">Health</a> <a href="/docs">Docs</a></div></div></body></html>"""

# ------------------------------------------------ Routes
@app.get("/")
async def root():
    p = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(p) and os.path.getsize(p) > 10000:
        with open(p,"r",encoding="utf-8") as f: return HTMLResponse(f.read())
    return HTMLResponse(HOLDING_PAGE.replace("{entries}",str(len(pnf_data))))

@app.get("/health")
async def health():
    return JSONResponse({"status":"ok","entries_loaded":len(pnf_data),
        "gemma_available":_GEMMA_MODEL is not None,"brave_available":bool(BRAVE_KEY) and _HAS_REQUESTS})

@app.post("/api/auth/register", response_model=AuthResponse)
async def register(r: AuthRequest):
    e = r.email.strip().lower()
    if not e or not ("@" in e and "." in e.split("@")[-1]):
        raise HTTPException(422, detail="Valid email required.")
    if len(r.password) < 6:
        raise HTTPException(422, detail="Password must be >= 6 chars.")
    if e in _users:
        raise HTTPException(409, detail="Email already registered. Sign in instead.")
    _users[e] = {"password_hash": _hash_pw(r_password), "created_at": time.time()}
    return AuthResponse(token=_make_token(e), email=e, message="Account created.")

@app.post("/api/auth/login", response_model=AuthResponse)
async def login(r: AuthRequest):
    e = r.email.strip().lower()
    u = _users.get(e)
    if not u: raise HTTPException(401, detail="Email not found.")
    if u["password_hash"] != _hash_pw(r.password):
        raise HTTPException(401, detail="Incorrect password.")
    return AuthResponse(token=_make_token(e), email=e, message="Signed in.")

@app.get("/api/auth/me")
async def me(authorization: Optional[str] = Header(None)):
    u = _resolve_token(authorization)
    if not u: raise HTTPException(401, detail="Not authenticated.")
    return u

@app.post("/api/pnf/ask", response_model=AskResponse)
async def ask(request: AskRequest, authorization: Optional[str] = Header(None)):
    user = _resolve_token(authorization)
    q = request.question.strip()
    if not q: raise HTTPException(422, detail="'question' must not be empty.")

    if is_interaction(q):
        drugs = [d.strip() for d in re.split(r" and | vs | versus ", q.lower()) if d.strip()]
        try:
            ai_html = _format_html(synthesize(drugs))
            return AskResponse(
                body=_ai_notice()+"\n"+ai_html,
                sources=[SourceItem(num=1,title="AI Synthesis",section=f"Interaction: {'+'„'.join(drugs)}",snippet="Generated by Gemini (Vertex AI). Always cross-check.",lastUpdated="Live")]
            )
        except Exception as err:
            return AskResponse(body=f"<p>Interaction synthesis failed: {str(err)[:100]}</p>",sources=[])

    match = _search_index(q)
    used_brave = False
    if match is None:
        g = brave_search_generic(q)
        if g:
            m2 = _search_index(g)
            if m2: match = m2; used_brave = True
    if match is None:
        return AskResponse(body=f"<p>No information found for <strong>{q}</strong> in the PNF index.</p>",sources=[])

    dn = match.get("drug",q)
    ct = _clean_text(match.get("text",""))
    parts = []
    if used_brave:
        parts.append(f'<p style="background:#f0f7ff;border-left:4px solid #6366f1;padding:0.6em 0.8em;border-radius:4px;margin-bottom:0.8em;font-size:0.9em"><strong>Brand &rarr; Generic:</strong> \"{q}\" resolved to <em>{dn}</em> via Brave Search.</p>')
    a = _ams_alert(dn)
    if a: parts.append(a)
    parts.append(_format_html(ct))
    parts.append(f"<p>{_citation(1,dn)}</p>")
    sn = ct[:200].strip()
    if len(ct)>200 and not ct[200].isspace():
        ls = sn.rfind(" ")
        if ls>0: sn = sn[:ls]
    sn += "\u2026" if len(ct)>200 else ""
    return AskResponse(body="\n�.join(parts),sources=[SourceItem(num=1,title="Philippine National Formulary",section=f"Drug Monograph \u2014 {dn}",snippet=sn,lastUpdated="Apr 2026")])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8501, reload=True)
