from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import re
import requests
import vertexai
from vertexai.generative_models import GenerativeModel
from dotenv import load_dotenv

# --- CONFIG ---
load_dotenv()
vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))
model = GenerativeModel(os.getenv("MODEL_NAME"))
BRAVE_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

AMS_RESTRICTED = ["cefepime", "ertapenem", "meropenem", "vancomycin", "amphotericin b", "voriconazole", "colistin", "micafungin", "aztreonam", "linezolid", "imipenem", "tigecycline"]

# --- APP ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "data", "pnf_index.json"), "r", encoding="utf-8") as f:
    pnf_data = json.load(f)

# --- MODELS ---
class AskRequest(BaseModel):
    question: str

# --- HELPERS ---
def _clean_text(raw):
    return re.sub(r"April.*?\n|https://.*?pnf\.doh\.gov\.ph\n+|ATC CODE\n+.*?\n+|Page \d of \d", "", raw)

def _search_index(query):
    q = query.lower().strip()
    for entry in pnf_data:
        if q == entry.get("drug", "").lower().strip(): return entry
    for entry in pnf_data:
        if q in entry.get("drug", "").lower().strip(): return entry
    return None

def brave_search_generic(brand_name):
    headers = {"X-Subscription-Token": BRAVE_KEY}
    res = requests.get(f"https://api.search.brave.com/res/v1/web/search?q={brand_name} generic name PNF", headers=headers)
    if res.status_code == 200:
        data = res.json()
        if "web" in data and "results" in data["web"]:
            return data["web"]["results"][0].get("title", "")
    return None

def synthesize_interaction(drugs):
    prompt = f"Provide a concise clinical interaction summary for: {', '.join(drugs)}"
    return model.generate_content(prompt).text

# --- ENDPOINTS ---
@app.get("/")
async def serve_frontend():
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/pnf/ask")
async def ask(request: AskRequest):
    q = request.question.strip()
    
    # 1. Interactions
    if " and " in q.lower() or " vs " in q.lower():
        drugs = [d.strip() for d in re.split(r' and | vs ', q.lower())]
        return JSONResponse({"body": synthesize_interaction(drugs)})

    # 2. PNF / Brand lookup
    match = _search_index(q)
    if not match:
        generic = brave_search_generic(q)
        if generic: match = _search_index(generic)
    
    if match:
        return JSONResponse({"body": _clean_text(match["text"])})
    
    return JSONResponse({"body": "<p>This query is outside the scope of the PNF.</p>"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
