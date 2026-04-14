import streamlit as st
import os
import requests
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader

# 1. Page Configuration
st.set_page_config(page_title="PNF Clinical Assistant", page_icon="💊")

# 2. API Key Loading (Railway Optimized)
# On Railway, we use os.getenv to read from the 'Variables' tab directly.
GROQ_KEY = os.getenv("GROQ_API_KEY")
BRAVE_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

with st.sidebar:
    st.header("⚙️ System Status")
    # If the variables aren't found, we show the manual input boxes
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

# 4. PNF Data Loading
@st.cache_resource
def load_pnf_context():
    # Use a relative path that works on Linux
    path = os.path.join("data", "PNF-Manual-for-Primary-Healthcare_8th.pdf")
    if os.path.exists(path):
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            return "\n".join([p.page_content for p in pages[50:150]])
        except Exception as e:
            return f"PDF Load Error: {e}"
    # If file is missing, the app should still RUN, not crash
    return "Clinical Manual currently unavailable (Check data folder)."

pnf_context = load_pnf_context()

# 5. Clinical UI
st.title("🇵🇭 PNF Clinical Assistant")
st.markdown("---")

user_query = st.text_input("Generic, Brand, or Combination:", placeholder="e.g. 'Biogesic' or 'Metronidazole + Azithromycin'")

if user_query:
    with st.spinner("Consulting PNF & Brave AI Context..."):
        # Check for combination queries
        is_combo = any(x in user_query.lower() for x in ["+", "and", "&", "interaction", "with"])
        
        # BRAVE SEARCH
        headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY}
        params = {"q": f"Philippine National Formulary PNF protocol {user_query}", "count": 3}
        
        try:
            search_resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            search_data = search_resp.json()
            results = search_data.get('web', {}).get('results', [])
            web_context = "\n".join([r.get('description', '') for r in results]) if results else "No additional context found."
        except Exception as e:
            web_context = f"Search failed: {e}"

        # SMART PROMPT
        prompt = f"""
        USER QUERY: {user_query}
        LOCAL PNF DATA: {pnf_context[:3000]}
        WEB SEARCH CONTEXT: {web_context}

        STRICT CLINICAL RULES:
        1. IF query involves TWO or MORE drugs:
           - IGNORE separate monographs.
           - Explain the SYNDROMIC MANAGEMENT or CLINICAL PROTOCOL (e.g. STI/PID).
           - Focus on the clinical reason for the combination and shared precautions.
        
        2. IF query is a SINGLE DRUG:
           - Provide: Generic Name, PNF Status, Dosage Form & Strengths, Indications, Contraindications.

        3. Always cite 'PNF 8th Edition'. Keep it professional and concise.
        """

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
