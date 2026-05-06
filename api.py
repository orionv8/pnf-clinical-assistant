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

import firebase_admin
from firebase_admin import credentials, auth, firestore

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

_FIRESTORE_DB = None
try:
    # Initialize Firebase Admin SDK using Application Default Credentials
    firebase_admin.initialize_app(credentials.ApplicationDefault())
    _FIRESTORE_DB = firestore.client()
    print("[Firebase] Admin SDK initialized using ADC.")
except Exception as e:
    print(f"[Firebase] Admin SDK initialization failed: {e}")

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
        _raw = _entry.get("drug", "")
        _drug = _raw.lower().strip().strip("\ufeff\u200b\r")
        _text = _entry.get("text", "")
        if not _drug:
            continue

        _clean = _clean_text(_text)
        _entry["clean_text"] = _clean

        drug_index[_drug] = _entry
        # Also index a fully-normalized version (letters+digits+spaces only)
        _norm = re.sub(r"[^a-z0-9 ]", "", _drug).strip()
        if _norm and _norm != _drug:
            drug_index[_norm] = _entry
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


def _search_index(query: str):
    q = query.lower().strip()
    if not q:
        return None

    words = re.findall(r"\b\w+\b", q)

    # 1. Exact match (original + normalized)
    if q in drug_index:
        return drug_index[q]
    q_norm = re.sub(r"[^a-z0-9 ]", "", q).strip()
    if q_norm != q and q_norm in drug_index:
        return drug_index[q_norm]

    # 2. Prefix match
    if q in prefix_index:
        return prefix_index[q][0]

    # 3. Single-word guard: block before fuzzy (prevents wrong matches)
    if len(words) == 1 and words[0] not in drug_index:
        if q_norm not in drug_index:
            return None

    # 4. Fuzzy match (>= 90 score — raised from 85 to reduce false positives)
    if drug_names and len(words) <= 3:
        best, score, _ = process.extractOne(q, drug_names, scorer=fuzz.WRatio)
        if score >= 90:
            return drug_index[best]

    # 5. Phrase + content search
    _stop = {"what","are","the","for","in","of","a","an","and","to","is","how",
             "does","do","can","with","this","that","from","by","on","at","or"}
    # Also filter generic words that match too many monographs
    _generic = {"first","line","second","third","use","used","drug","dose",
                "treatment","treatments","adults","adult","children","patient","patients"}
    mw = [w for w in words if w not in _stop and len(w) >= 3]
    # Specific medical words (not generic) — used for scoring
    specific = [w for w in mw if w not in _generic]

    # Phrase match: prioritize SPECIFIC medical phrases
    if len(mw) >= 2 and specific:
        hits = []
        for e in pnf_data:
            tn = re.sub(r"[^a-z0-9 ]", " ", e.get("clean_text","").lower())
            # Check if any specific term appears in text
            if not any(w in tn for w in specific):
                continue
            # Check for consecutive phrase pairs
            for i in range(len(mw)-1):
                if mw[i]+" "+mw[i+1] in tn:
                    hits.append(e); break
        if hits:
            # Score by SPECIFIC word matches (not generic ones)
            hits.sort(key=lambda e: sum(1 for w in specific if w in
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
        unique = list({c["drug"]: c for c in cands}.values())
        return sorted(unique, key=_sc, reverse=True)[0]

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
    # Try normalized (strip hyphens, special chars)
    tn = re.sub(r"[^a-z0-9 ]", "", t).strip()
    if tn != t and tn in drug_index:
        return drug_index[tn], "none", None
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
    label = "MIMS brand database" if source == "mims" else "PNF Assistant AI"
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

def _build_ai_notice() -> str:
    return (
        '<p class="ai-notice" style="'
        "background:#e8f0fe;border-left:4px solid #4285f4;"
        "padding:0.6em 0.8em;border-radius:4px;margin-bottom:0.8em;"
        'font-size:0.9em;">'
        "<strong>&#10024; AI-Synthesized Summary:</strong> Generated by PNF Assistant AI. "
        "Always cross-check with PNF and authoritative sources."
        "</p>"
    )

def synthesize_interaction(drugs: List[str], is_question: bool = False, full_query: str = "") -> str:
    if _GEMMA_MODEL is None:
        raise RuntimeError("Gemini not configured (set PROJECT_ID, LOCATION, MODEL_NAME env vars).")
    
    if is_question:
        prompt = (
            f"Answer the following clinical question based strictly on standard medical guidelines and the Philippine National Formulary context:\n\n"
            f"Question: {full_query}\n\n"
            "Provide a concise, professional clinical response in plain text. "
            "Do not include disclaimers — they are added by the UI."
        )
    else:
        prompt = (
            "Provide a concise clinical drug interaction summary for the following "
            "medications: "
            + ", ".join(drugs)
            + ". Cover mechanism, severity, and clinical management in plain text. "
            "Do not include disclaimers — they are added by the UI."
        )
    return _GEMMA_MODEL.generate_content(prompt).text


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
        "version": "3.3.1",
        "entries_loaded": len(pnf_data),
        "optimized_search": True,
        "gemma_available": _GEMMA_MODEL is not None,
    }
    return {**base, **get_mims_status()}


@app.post("/api/auth/register")
async def register(req: AuthRequest):
    if _FIRESTORE_DB is None:
        raise HTTPException(status_code=500, detail="Firebase not initialized")
    try:
        user = auth.create_user(email=req.email.strip().lower(), password=req.password)
        # Optionally, save additional user data to Firestore
        # _FIRESTORE_DB.collection("users").document(user.uid).set({"email": user.email})
        custom_token = auth.create_custom_token(user.uid).decode('utf-8')
        return {"token": custom_token, "email": user.email}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/login")
async def login(req: AuthRequest):
    if _FIRESTORE_DB is None:
        raise HTTPException(status_code=500, detail="Firebase not initialized")
    try:
        # Authenticate via custom token for existing users
        user = auth.get_user_by_email(req.email.strip().lower())
        # For login, Firebase Admin SDK doesn't directly expose password verification.
        # The client-side SDK handles email/password login and provides an ID token.
        # For this backend, we'll create a custom token to allow client to sign in.
        custom_token = auth.create_custom_token(user.uid).decode('utf-8')
        return {"token": custom_token, "email": user.email}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/auth/me")
async def me(authorization: Optional[str] = Header(None)):
    if _FIRESTORE_DB is None:
        raise HTTPException(status_code=500, detail="Firebase not initialized")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    id_token = authorization[7:].strip()
    try:
        decoded_token = auth.verify_id_token(id_token)
        return {"email": decoded_token["email"]}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {e}")


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
    is_interaction = len(entities) > 1

    # --- Step 3: Handle Questions and Interactions (AI Synthesis) ---
    if is_question or is_interaction:
        try:
            ai_text = synthesize_interaction(drugs=entities, is_question=is_question, full_query=question)
            ai_html = _format_text_as_html(ai_text)
            body_html = "\n".join([_build_ai_notice(), ai_html])
            
            title_text = "AI-Synthesized Answer" if is_question else "AI-Synthesized Summary"
            section_text = f"Clinical Question" if is_question else f"Drug Interaction: {' + '.join(entities)}"

            return AskResponse(
                body=body_html,
                sources=[SourceItem(
                    num=1,
                    title=title_text,
                    section=section_text,
                    snippet="Generated by PNF Assistant AI. Always cross-check with PNF and authoritative sources.",
                    lastUpdated="Live",
                )],
            )
        except Exception as e:
            err_html = (
                f"<p>Unable to synthesize response: {str(e)[:120]}</p>"
                "<p>Try searching specific drug names in the PNF library instead.</p>"
            )
            return AskResponse(body=err_html, sources=[])

    # --- Step 4: Resolve each entity independently ---
    results = []  # list of (entity, match, resolver, generic)
    for ent in entities:
        ew = re.findall(r"\b\w+\b", ent)
        is_brand = len(ew) <= 3 and not is_question

        # 3a. Direct resolve (exact + MIMS)
        m, rv, gen = _resolve_one(ent)

        # 3b. Gemini fallback (allow any length now that hallucinations are fixed)
        if m is None and is_brand:
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
