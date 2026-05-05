# PNF Clinical Assistant

Philippine National Formulary drug reference chatbot.

## Architecture

```
index.html (chatbot UI)
  ↓ POST /api/pnf/ask
api.py (FastAPI)
  ↓ 1. PNF keyword search (pnf_index.json, 746 drugs)
  ↓ 2. MIMS brand→generic lookup (mims_brand_generic_names.txt)
  ↓ 3. Brave Search fallback (brand→generic via web)
  ↓ 4. Fuzzy search (rapidfuzz)
  ↓ PNF text reference ONLY (no AI-generated drug info)
```

Gemini is used ONLY for drug interaction queries ("X and Y").

## Required Environment Variables (Cloud Run)

| Variable | Value | Notes |
|----------|-------|-------|
| `PROJECT_ID` | `pnf-clinical-assistant-app` | GCP project |
| `LOCATION` | `global` | Vertex AI region |
| `MODEL_NAME` | `google/gemma-4-26b-a4b-it-maas` | Gemma MaaS model |
| `BRAVE_SEARCH` | `BSArvn...` | Brave Search API key |

**DO NOT change MODEL_NAME or LOCATION without documenting the original values.**

## Local Development

```bash
pip install -r requirements.txt
python api.py  # Runs on port 8501
```

## Deploy to Cloud Run

Push to `main` auto-deploys via Cloud Build trigger.

Manual deploy:
```bash
gcloud run deploy pnf-clinical-assistant \
  --source . --region asia-southeast1 --allow-unauthenticated
```

## Key Files

| File | Purpose |
|------|---------|
| `api.py` | FastAPI backend (main) |
| `ai_resolver.py` | MIMS brand→generic lookup |
| `brave_resolver.py` | Brave Search brand→generic fallback |
| `index.html` | Chatbot frontend |
| `data/pnf_index.json` | PNF drug index (746 entries) |
| `data/mims_brand_generic_names.txt` | Brand→generic mapping |
| `Dockerfile` | Container build (runs uvicorn) |

## NOT in use (legacy)

| File | Status |
|------|--------|
| `app.py` | Old Streamlit app — replaced by api.py |
| `simple_server.py` | Old static server — replaced by api.py |
