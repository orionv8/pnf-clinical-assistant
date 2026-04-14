import streamlit as st
import os
import requests
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader

# 1. Page Configuration
st.set_page_config(page_title="PNF Clinical Assistant", page_icon="💊")

# 2. API Key Loading (Railway Optimized)
GROQ_KEY = os.getenv("GROQ_API_KEY")
BRAVE_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

with st.sidebar:
    st.header("⚙️ System Status")
    if not GROQ_KEY or not BRAVE_KEY:
        st.error("Keys missing in Railway Variables")
        GROQ_KEY = st.text_input("Manual Groq API Key", value=GROQ_KEY if GROQ_KEY else "", type="password")
        BRAVE_KEY = st.text_input("Manual Brave API Key", value=BRAVE_KEY if BRAVE_KEY else "", type="password")
    else:
        st.success("✅ PNF Engine Connected")

if not (GROQ_KEY and BRAVE_KEY):
    st.info("Awaiting API Keys to initialize...")
    st.stop()

# 3. Initialize Groq
groq_client = Groq(api_key=GROQ_KEY)

# 4. PNF Data Loading (Cached for Speed)
@st.cache_resource
def load_pnf_context():
    path = os.path.join("data", "PNF-Manual-for-Primary-Healthcare_8th.pdf")
    if os.path.exists(path):
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            # Focusing on protocol-heavy pages
            return "\n".join([p.page_content for p in pages[50:200]])
        except Exception as e:
            return f"Error loading PDF: {e}"
    return "Local PNF Manual not found."

pnf_context = load_pnf_context()

# 5. Clinical UI
st.title("🇵🇭 PNF Clinical Assistant")
st.markdown("---")

user_query = st.text_input("Generic, Brand, or Combination:", placeholder="e.g. 'Ceftriaxone' or 'Metronidazole + Azithromycin'")

if user_query:
    with st.spinner("Consulting PNF Protocols..."):
        # Detect if it's a combo or interaction query
        is_combo = any(x in user_query.lower() for x in ["+", "and", "&", "interaction", "with", "vs"])
        
        # BRAVE SEARCH
        headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY}
        params = {"q": f"Philippine National Formulary PNF protocol {user_query}", "count": 3}
        
        try:
            search_resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            search_data = search_resp.json()
            results = search_data.get('web', {}).get('results', [])
            web_context = "\n".join([r.get('description', '') for r in results])
        except:
            web_context = "Web search unavailable."

        # THE "NO-FLUFF" CLINICAL PROMPT
        prompt = f"""
        USER QUERY: {user_query}
        LOCAL PNF DATA: {pnf_context[:4000]}
        WEB SEARCH: {web_context}

        STRICT INSTRUCTIONS:
        - YOU ARE A CLINICAL PHARMACIST. BE DIRECT.
        
        1. IF THE QUERY IS A COMBINATION OR INTERACTION (e.g., A + B):
           - **STRICT RULE:** DO NOT list individual monographs, classifications, or dosages for each drug separately.
           - **ACTION:** Jump IMMEDIATELY to the '📋 Clinical Summary'.
           - Explain the 'Syndromic Management' or 'Clinical Protocol' (e.g., PID, STI, etc.) where these drugs are used together.
           - Detail the combined regimen (e.g., 'Drug A 500mg BID + Drug B 1g single dose').
           - Highlight 'Key PNF Precautions' (e.g., Alcohol avoidance, QTc prolongation).
           - Do not give any other background info.

        2. IF THE QUERY IS A SINGLE DRUG:
           - Follow the standard Monograph template:
             - "Based strictly on the PNF 8th Edition..."
             - ### 1. Formulary Status
             - ### 2. Clinical Monograph
           
        Always cite 'PNF 8th Edition'.
        """

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a direct Clinical Pharmacist. If a combination is asked, you answer ONLY with the protocol/interaction. No separate drug summaries allowed."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=850
            )
            st.markdown("---")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Groq Error: {e}")
