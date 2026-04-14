import streamlit as st '
import os
import requests
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader

# 1. Page Configuration
st.set_page_config(page_title="PNF Clinical Assistant", page_icon="💊")

# 2. API Key Loading (Resilient for Cloud Deployment)
# This checks Railway's Environment Variables first, then falls back to secrets.toml
GROQ_KEY = os.getenv("GROQ_API_KEY") or (st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else None)
BRAVE_KEY = os.getenv("BRAVE_SEARCH_API_KEY") or (st.secrets.get("BRAVE_SEARCH_API_KEY") if "BRAVE_SEARCH_API_KEY" in st.secrets else None)

with st.sidebar:
    st.header("⚙️ System Status")
    if not GROQ_KEY or not BRAVE_KEY:
        st.error("API Keys missing in Railway Variables")
        # Backup manual input for iPad
        GROQ_KEY = st.text_input("Manual Groq API Key", value=GROQ_KEY if GROQ_KEY else "", type="password")
        BRAVE_KEY = st.text_input("Manual Brave API Key", value=BRAVE_KEY if BRAVE_KEY else "", type="password")
    else:
        st.success("✅ PNF Engine Connected")

if not (GROQ_KEY and BRAVE_KEY):
    st.info("Awaiting API Keys to initialize...")
    st.stop()

# 3. Initialize Groq
groq_client = Groq(api_key=GROQ_KEY)

# 4. PNF Data Loading (Focused Snippet for Speed)
@st.cache_resource
def load_pnf_context():
    path = "data/PNF-Manual-for-Primary-Healthcare_8th.pdf"
    if os.path.exists(path):
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            # Focusing on the middle section where common clinical protocols live
            return "\n".join([p.page_content for p in pages[50:150]])
        except Exception as e:
            return f"Error loading PDF: {e}"
    return "Local PNF Manual not found in /data folder."

pnf_context = load_pnf_context()

# 5. Clinical UI
st.title("🇵🇭 PNF Clinical Assistant")
st.markdown("---")

user_query = st.text_input("Generic, Brand, or Combination:", placeholder="e.g. 'Biogesic' or 'Metronidazole + Azithromycin'")

if user_query:
    with st.spinner("Consulting PNF & Brave AI Context..."):
        # Detect Combination/Interaction
        is_combo = any(x in user_query.lower() for x in ["+", "and", "&", "interaction", "with"])
        
        # --- BRAVE SEARCH CALL ---
        headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY}
        params = {"q": f"Philippine National Formulary PNF protocol {user_query}", "count": 3}
        
        try:
            search_resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            search_data = search_resp.json()
            results = search_data.get('web', {}).get('results', [])
            web_context = "\n".join([r.get('description', '') for r in results]) if results else "No additional web context found."
        except Exception as e:
            web_context = f"Search failed: {e}"

        # --- SMART PROMPT ---
        prompt = f"""
        USER QUERY: {user_query}
        LOCAL PNF DATA: {pnf_context[:3000]}
        WEB SEARCH CONTEXT: {web_context}

        STRICT CLINICAL RULES:
        1. IF query involves TWO or MORE drugs:
           - IGNORE separate monographs/dosages.
           - Explain the SYNDROMIC MANAGEMENT or CLINICAL PROTOCOL (e.g. STI/PID).
           - Focus on WHY they are combined and shared precautions.
        
        2. IF query is a SINGLE DRUG:
           - Provide: Generic Name, PNF Status, Dosage Form & Strengths, Indications, Contraindications.

        3. Always cite 'PNF 8th Edition'. Keep it professional and concise.
        """

        # --- GROQ AI RESPONSE ---
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a Clinical Pharmacist in the Philippines. You provide protocol-based answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=750
            )
            st.markdown("---")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Groq Error: {e}")
