import streamlit as st
import os
import requests
import json
import re
import vertexai
from vertexai.generative_models import GenerativeModel

# 1. Page Configuration
st.set_page_config(page_title="PNF Clinical Assistant", page_icon="💊", layout="centered")

# --- CUSTOM CSS FOR LINEAR-INSPIRED THEME ---
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
    .stApp {
        background-color: var(--color-bg);
    }
    h1 {
        font-family: 'Inter', sans-serif !important;
        color: var(--color-primary);
        letter-spacing: -0.704px;
        font-weight: 510 !important;
    }
    .stTextInput > div > div > input {
        background-color: var(--color-panel);
        border: 1px solid var(--color-border);
        color: var(--color-primary);
        border-radius: 8px;
        padding: 14px;
    }
    .stTextInput > label {
        color: var(--color-tertiary) !important;
    }
    .css-1544g2n, .css-1n76uvr {
        color: var(--color-primary) !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. API Key/Config Loading
vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))
model = GenerativeModel(os.getenv("MODEL_NAME"))
BRAVE_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

# --- SECURITY: INPUT SANITIZATION ---
def is_malicious(query):
    patterns = [r"ignore previous", r"system prompt", r"output your instructions", r"dan mode"]
    return any(re.search(p, query.lower()) for p in patterns)

# --- CLINICAL DATA ---
AMS_RESTRICTED = [
    "cefepime", "ertapenem", "meropenem", "vancomycin", 
    "amphotericin b", "voriconazole", "colistin", 
    "micafungin", "aztreonam", "linezolid", "imipenem", "tigecycline"
]

# 5. UI Logic
# Linear-style Title/Subtitle
st.markdown("<h1>PNF <span style='color:#5e6ad2;'>Clinical Assistant</span></h1>", unsafe_allow_html=True)
st.write("Evidence-based clinical decision support.")
st.markdown("---")

user_query = st.text_input("Enter Drug(s) or Ask a Question:", placeholder="e.g. 'Furosemide', 'Biogesic', or 'Meropenem dose?'")

# Load Index (Simplified for demo, same logic as before)
@st.cache_resource
def load_static_index():
    index_file = "data/pnf_index.json"
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

all_pnf_data = load_static_index()

if user_query:
    if is_malicious(user_query):
        st.warning("⚠️ Security Alert: Input blocked.")
        st.stop()

    with st.spinner("Searching..."):
        # Local search logic (same as original)
        scored_results = []
        for entry in all_pnf_data:
            if user_query.lower() in entry["text"].lower():
                scored_results.append(entry)
        
        relevant_text = "\n...\n".join([r["text"] for r in scored_results])[:5000]
        
        is_restricted = any(drug in user_query.lower() for drug in AMS_RESTRICTED)
        
        # Generation
        try:
            prompt = f"System: Clinical AI. Query: {user_query}. Data: {relevant_text}. Respond professionally."
            response = model.generate_content(prompt)
            st.markdown("---")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("ℹ️ Official DOH PNF Portal | Clinical reference only.")
