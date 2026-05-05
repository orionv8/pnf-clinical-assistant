# api.py — PNF Clinical Assistant API v3.3.0
# MIMS brand resolver + Gemini AI fallback + auth + AMS alerts

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

from ai_resolver import ai_resolve_generic

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError: pass

_GEMMA_MODEL = None
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    _p, _l, _m = os.getenv("PROJECT_ID"), os.getenv("LOCATION"), os.getenv("MODEL_NAME")
    if _p and _l and _m:
        vertexai.init(project=_p, location=_l)
        _GEMMA_MODEL = GenerativeModel(_m)
except Exception: pass

app = FastAPI(title="PNF Clinical Assistant API", version="3.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AMS_RESTRICTED = [
    "cefepime", "ertapenem", "meropenem", "vancomycin",
    "amphotericin b", "voriconazole", "colistin", "micafungin",
    "aztreonam", "linezolid", "imipenem", "tigecycline",
]

pnf_data: list = []
drug_index: dict = {}
prefix_index: dict = {}
content_index: dict = {}
drug_names: list = []


def _clean_text(raw: str) -> str:
    cleaned = raw.lstrip("\ufeff")
    return re.sub(
        r"April.*?\n|https://.*?pnf\.doh\.gov\.ph\n+|ATC CODE\n+.*?\n+|Page \d of \d",
        "",
        cleaned,
    )


index_path = os.path.join(BASE_DIR, "data", "pnf_index.json")

if os.path.exists(index_path):
    with open(index_path, "r", encoding="utf-8") as _f:
        pnf_data = json.load(_f)

    for _entry in pnf_data:
        _drug = _entry.get("drug", "").lower().strip()
        _text = _entry.get("text", "")
        if not _drug:
            continue

        _clean = _clean_text(_text)
        _entry["clean_text"] = _clean

        drug_index[_drug] = _entry
        drug_names.append(_drug)

        # prefix index (up to 10 chars)
        for _i in range(1, min(len(_drug), 10) + 1):
            prefix_index.setdefault(_drug[:_i], []).append(_entry)

        # inverted content index (words > 3 chars)
        for _w in set(re.findall(r"\b\w+\b", _clean.lower())):
            if len(_w) > 3:
                content_index.setdefault(_w, []).append(_entry)

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

_users: dict = {}
_tokens: dict = {}


def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _create_token(email: str) -> str:
    tok = secrets.token_urlsafe(32)
    _tokens[tok] = {"email": email, "expires": time.time() + 365 * 86400}
    return tok


def _resolve_token(auth: Optional[str]) -> Optional[dict]:
    if not auth or not auth.startswith("Bearer "):
        return None
    tok = auth[7:].strip()
    rec = _tokens.get(tok)
    if not rec or time.time() > rec["expires"]:
        return None
    return {"email": rec["email"]}


@lru_cache(maxsize=500)
def _search_index(query: str):
    q = query.lower().strip()
    if not q:
        return None

    words = re.findall(r"\b\w+\b", q)

    # 1. Exact match
    if q in drug_index:
        return drug_index[q]

    # 2. Prefix match
    if q in prefix_index:
        return prefix_index[q][0]

    # 3. Fuzzy match (>= 85 score)
    if drug_names:
        best, score, _ = process.extractOne(q, drug_names, scorer=fuzz.WRatio)
        if score >= 85:
            return drug_index[best]

    # 4. Block single brand-like words from falling into content search
    if len(words) == 1 and words[0] not in drug_index:
        return None

    # 5. Phrase + content search with medical-term weighting
    _stop = {"what","are","the","for","in","of","a","an","and","to","is","how",
             "does","do","can","with","this","that","from","by","on","at","or"}
    mw = [w for w in words if w not in _stop and len(w) >= 3]

    # Phrase match: find entries containing consecutive medical terms
    if len(mw) >= 2:
        hits = []
        for e in pnf_data:
            tn = re.sub(r"[^a-z0-9 ]", " ", e.get("clean_text","").lower())
            for i in range(len(mw)-1):
                if mw[i]+" "+mw[i+1] in tn:
                    hits.append(e); break
        if hits:
            hits.sort(key=lambda e: sum(1 for w in mw if w in
                      re.sub(r"[^a-z0-9 ]"," ",e.get("clean_text","").lower())), reverse=True)
            return hits[0]

    # Inverted-index fallback
    cands = []
    for w in mw:
        if len(w) >= 4 and w in content_index:
            cands.extend(content_index[w])
    if cands:
        def _sc(e):
            s, d, t = 0, e["drug"].lower(), e.get("clean_text","").lower()
            if q == d: s += 100
            elif q in d: s += 60
            h = sum(1 for w in mw if w in t)
            return s + (h*10 if h >= 2 else 0)
        return sorted(set(cands), key=_sc, reverse=True)[0]

    return None


def _format_text_as_html(text: str) -> str:
    parts = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        core = s.rstrip(":.")
        if len(core) >= 3 and core == core.upper() and core.replace(" ", "").isalpha():
            parts.append(f"<p><strong>{s}</strong></p>")
        else:
            parts.append(f"<p>{s}</p>")
    return "\n".join(parts)


def _build_ams_alert(drug_name: str) -> str:
    if drug_name.lower().strip() in AMS_RESTRICTED:
        return (
            '<p class="ams-alert" style="'
            "background:#fff3cd;border-left:4px solid #ffc107;"
            "padding:0.6em 0.8em;border-radius:4px;margin-bottom:0.8em;"
            '">'
            "<strong>&#9888; AMS Restricted Antimicrobial</strong> &mdash; "
            f"<em>{drug_name}</em> is subject to Antimicrobial Stewardship "
            "Programme (AMS) restrictions. Use requires documented indication "
            "and, where applicable, Infectious Disease specialist approval."
            "</p>"
        )
    return ""


def _resolve_one(term: str):
    """Resolve a single drug term: exact → MIMS → fuzzy. Returns (match, resolver, generic) or (None,None,None)."""
    t = term.lower().strip()
    if t in drug_index:
        return drug_index[t], "none", None
    from ai_resolver import MIMS_BRAND_TO_GENERIC
    mk = term.strip().upper()
    mh = MIMS_BRAND_TO_GENERIC.get(mk) or (MIMS_BRAND_TO_GENERIC.get(mk.split()[0]) if " " in mk else None)
    if mh and mh.lower().strip() in drug_index:
        return drug_index[mh.lower().strip()], "mims", mh.lower().strip()
    m = _search_index(term)
    return (m, "none", None) if m else (None, None, None)


def _build_citation(num: int, drug_name: str) -> str:
    return (
        f'<a class="citation" href="#source-{num}" '
        f'title="Philippine National Formulary &mdash; {drug_name}">'
        f"[{num}]</a>"
    )


def _resolver_notice(brand: str, generic: str, source: str = "MIMS") -> str:
    label = "MIMS brand database" if source == "mims" else "Gemini AI"
    return (
        '<p class="resolver-notice" style="'
        "background:#e6ffed;border-left:4px solid #4ade80;"
        "padding:0.6em 0.8em;border-radius:4px;margin-bottom:0.8em;"
        'font-size:0.9em;">'
        f"<strong>Brand &rarr; Generic:</strong> "
        f'"{brand}" was resolved to <em>{generic.upper()}</em> '
        f"via {label}. Showing PNF data for the generic."
        "</p>"
    )


# Routes
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    html_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(html_path, "r", encoding="utf-8") as fh:
        return HTMLResponse(content=fh.read())


@app.get("/health")
async def health():
    from ai_resolver import get_mims_status
    base = {
        "status": "ok",
        "version": "3.2.0",
        "entries_loaded": len(pnf_data),
        "optimized_search": True,
        "gemma_available": _GEMMA_MODEL is not None,
    }
    return {**base, **get_mims_status()}


@app.post("/api/auth/register")
async def register(req: AuthRequest):
    email = req.email.strip().lower()
    if not email or not req.password:
        raise HTTPException(status_code=422, detail="Email and password required")
    if email in _users:
        raise HTTPException(status_code=409, detail="Email already registered")
    _users[email] = {"hash": _hash_password(req.password)}
    token = _create_token(email)
    return {"token": token, "email": email}


@app.post("/api/auth/login")
async def login(req: AuthRequest):
    email = req.email.strip().lower()
    user = _users.get(email)
    if not user or user["hash"] != _hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = _create_token(email)
    return {"token": token, "email": email}


@app.get("/api/auth/me")
async def me(authorization: Optional[str] = Header(None)):
    user = _resolve_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@app.post("/api/pnf/ask", response_model=AskResponse)
async def ask(req: AskRequest, authorization: Optional[str] = Header(None)):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Empty query")

    q = question.lower().strip()
    q_words = re.findall(r"\b\w+\b", q)

    # --- Step 1: Extract entities (split multi-drug queries) ---
    # "digoxin and furosemide" → ["digoxin", "furosemide"]
    # "biogesic" → ["biogesic"]
    _splitters = [" and ", " with ", " vs ", " versus ", ", "]
    entities = [q]
    for sp in _splitters:
        if sp in q:
            entities = [p.strip() for p in q.split(sp.strip()) if p.strip()]
            break

    # --- Step 2: Classify query ---
    _qw = {"what","how","which","when","where","why","can","does",
            "should","list","tell","give","compare"}
    is_question = len(q_words) > 4 or (q_words and q_words[0] in _qw)

    # --- Step 3: Resolve each entity independently ---
    results = []  # list of (entity, match, resolver, generic)
    for ent in entities:
        ew = re.findall(r"\b\w+\b", ent)
        is_brand = len(ew) <= 3 and not is_question

        # 3a. Direct resolve (exact + MIMS)
        m, rv, gen = _resolve_one(ent)

        # 3b. Gemini fallback (only 2-3 word brands)
        if m is None and is_brand and len(ew) >= 2:
            g = ai_resolve_generic(ent, _GEMMA_MODEL)
            if g and g.lower().strip() in drug_index:
                m, rv, gen = drug_index[g.lower().strip()], "gemini", g.lower().strip()

        if m:
            results.append((ent, m, rv, gen))

    # --- Step 4: Fallback — phrase/content search on full query ---
    if not results:
        m = _search_index(question)
        if m:
            results.append((question, m, "none", None))

    # --- Step 5: Not found ---
    if not results:
        return AskResponse(
            body=f"<p>No information found for <strong>{question}</strong> in the "
                 "Philippine National Formulary. Try a generic drug name.</p>",
            sources=[])

    # --- Step 6: Build combined response ---
    body_parts = []
    sources = []
    for i, (ent, match, resolver, gen) in enumerate(results):
        dn = match.get("drug", ent)
        ct = match.get("clean_text", "")
        notice = ""
        if resolver in ("mims","gemini") and gen:
            notice = _resolver_notice(ent, gen, source=resolver)
        ams = _build_ams_alert(dn)
        cite = _build_citation(i+1, dn)
        body_parts.append(notice + ams + _format_text_as_html(ct) + f"\n<p>{cite}</p>")
        snip = ct[:200].strip()
        if len(ct) > 200:
            sp = snip.rfind(" ")
            snip = (snip[:sp] if sp > 0 else snip) + "..."
        sources.append(SourceItem(num=i+1, title="Philippine National Formulary",
            section=f"Drug Monograph &mdash; {dn}", snippet=snip, lastUpdated="Apr 2026"))

    sep = '<hr style="margin:1.5em 0;border:none;border-top:1px solid #ddd;">'
    return AskResponse(body=sep.join(body_parts), sources=sources)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8501, reload=True)
