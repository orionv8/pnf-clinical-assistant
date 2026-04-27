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

# --- UI LOGIC ---
st.markdown('<div class="main-title">Search the Formulary</div>', unsafe_allow_html=True)
user_query = st.text_input("", placeholder="Enter drug name or clinical question...")

# Cards
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="card"><h3>About PNF</h3><p>The Philippine National Formulary (PNF) is the essential list of medicines for the Philippine healthcare system, ensuring safety, efficacy, and cost-effectiveness.</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card"><h3>Latest Updates</h3><p>Stay informed with the latest additions, removals, and clinical guidelines updates affecting the PNF and hospital formulary scenes.</p></div>', unsafe_allow_html=True)

# Logic
if user_query:
    with st.spinner("Searching..."):
        try:
            prompt = f"System: Clinical AI. Query: {user_query}. Respond professionally using PNF context."
            response = model.generate_content(prompt)
            st.markdown("---")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error: {e}")
