import streamlit as st
import os
import requests
import json
import re
import vertexai
from vertexai.generative_models import GenerativeModel

# 1. Page Configuration
st.set_page_config(page_title="PNF Clinical Assistant", page_icon="💊", layout="centered")

# --- MODERN MINIMALIST DESIGN (CSS INJECTION) ---
st.markdown("""
<style>
    :root {
        --color-bg: #08090a;
        --color-panel: #0f1011;
        --color-primary: #f7f8f8;
        --color-tertiary: #8a8f98;
        --color-accent: #5e6ad2;
        --color-border: rgba(255,255,255,0.08);
    }
    .stApp { background-color: var(--color-bg); }
    .main-title { font-size: 48px; font-weight: 510; letter-spacing: -1.056px; color: var(--color-primary); text-align: center; margin-bottom: 24px; }
    .stTextInput > div > div > input {
        background-color: var(--color-panel);
        border: 1px solid var(--color-border);
        color: var(--color-primary);
        padding: 20px;
        border-radius: 12px;
        font-size: 18px;
    }
    .card-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 40px; }
    .card { background: var(--color-panel); border: 1px solid var(--color-border); padding: 24px; border-radius: 12px; }
    .card h3 { color: var(--color-accent); margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# 2. API Key/Config
vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))
model = GenerativeModel(os.getenv("MODEL_NAME"))
BRAVE_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

# --- CLINICAL DATA ---
AMS_RESTRICTED = ["cefepime", "ertapenem", "meropenem", "vancomycin", "amphotericin b", "voriconazole", "colistin", "micafungin", "aztreonam", "linezolid", "imipenem", "tigecycline"]

# --- SMART SEARCH ENGINE ---
def search_local_index(query, index_data):
    # Split query into keywords
    keywords = [w for w in re.sub(r'[^\w\s]', '', query.lower()).split() if len(w) > 2]
    if not keywords: return []
    
    scored_results = []
    for entry in index_data:
        content = entry["text"].lower()
        score = sum(content.count(k) for k in keywords)
        if score > 0:
            scored_results.append({"text": entry["text"], "score": score})
    
    # Sort by relevance
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results

# --- UI LOGIC & CLEAR BUTTON ---
st.markdown("""
<style>
    div[data-testid="stTextInput"] > div > div > div > button { display: block !important; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">Search the Formulary</div>', unsafe_allow_html=True)
user_query = st.text_input("", placeholder="Enter drug name or clinical question...")

# Cards
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="card"><h3>About PNF</h3><p>The Philippine National Formulary (PNF) is the essential list of medicines for the Philippine healthcare system, ensuring safety, efficacy, and cost-effectiveness.</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card"><h3>Latest Updates</h3><p>Stay informed with the latest additions, removals, and clinical guidelines updates affecting the PNF and hospital formulary scenes.</p></div>', unsafe_allow_html=True)

# Load Index
@st.cache_resource
def load_static_index():
    index_file = "data/pnf_index.json"
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

all_pnf_data = load_static_index()

# Logic
if user_query:
    with st.spinner("Searching..."):
        # Local search logic
        scored_results = []
        for entry in all_pnf_data:
            if user_query.lower() in entry["text"].lower():
                scored_results.append(entry)
        
        relevant_text = "\n...\n".join([r["text"] for r in scored_results])[:5000]

        # --- AMS LOGIC ---
        is_restricted = any(drug in user_query.lower() for drug in AMS_RESTRICTED)
        ams_instruction = ""
        if is_restricted:
            ams_instruction = "\n### ⚠️ AMS ALERT: RESTRICTED ANTIMICROBIAL\n> **Note:** This medicine is a RESTRICTED antimicrobial. Usage requires institutional AMS clearance and specific justification."

        # Generation
        try:
            # Web Search Fallback
            web_context = ""
            if len(scored_results) < 2:
                headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY}
                params = {"q": f"{user_query} philippines", "count": 3}
                resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
                web_results = resp.json().get('web', {}).get('results', [])
                web_context = "\n".join([f"[WEB SOURCE: {r['url']}]\n{r['description']}" for r in web_results])

            prompt = f"System: Clinical AI. Query: {user_query}. Data: {relevant_text}. Web: {web_context}. Respond professionally. {ams_instruction}"
            response = model.generate_content(prompt)
            st.markdown("---")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error: {e}")
