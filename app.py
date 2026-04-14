import streamlit as st
import os
import requests
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader

# 1. Page Configuration
st.set_page_config(page_title="PNF Clinical Assistant", page_icon="💊")

# 2. API Key Loading (Brave + Groq)
GROQ_KEY = st.secrets.get("GROQ_API_KEY")
BRAVE_KEY = st.secrets.get("BRAVE_SEARCH_API_KEY")

with st.sidebar:
    st.header("⚙️ System Status")
    if not GROQ_KEY or not BRAVE_KEY:
        st.error("Keys missing in secrets.toml")
        GROQ_KEY = st.text_input("Groq API Key", type="password")
        BRAVE_KEY = st.text_input("Brave API Key", type="password")
    else:
        st.success("✅ PNF & Brave Integrated")

if not (GROQ_KEY and BRAVE_KEY):
    st.info("Awaiting API Keys...")
    st.stop()

# 3. Initialize Groq (Brave uses 'requests')
groq_client = Groq(api_key=GROQ_KEY)

# 4. PNF Data Loading (Focused Snippet for Speed)
@st.cache_resource
def load_pnf_context():
    path = "data/PNF-Manual-for-Primary-Healthcare_8th.pdf"
    if os.path.exists(path):
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            # Grabbing pages 50-150 where many STI/Common protocols live
            return "\n".join([p.page_content for p in pages[50:150]])
        except Exception as e:
            return f"Error loading PDF: {e}"
    return "Local PNF Manual not found."

pnf_context = load_pnf_context()

# 5. Clinical UI
st.title("🇵🇭 PNF Clinical Assistant")
st.markdown("---")

# FIX: We define the variable HERE before using it in the 'if' statement
user_query = st.text_input("Query (e.g., 'Metformin' or 'Metronidazole + Azithromycin'):")

if user_query:
    with st.spinner("Consulting PNF & Brave AI Context..."):
        # Detect Combination/Interaction
        is_combo = any(x in user_query.lower() for x in ["+", "and", "&", "interaction", "with"])
        
        # --- BRAVE SEARCH CALL ---
        headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY}
        # We use the 'web' search or 'llm/context' endpoint
        params = {"q": f"Philippine National Formulary PNF protocol {user_query}", "count": 3}
        
        try:
            search_resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            search_data = search_resp.json()
            # Extract snippets from search results
            web_context = "\n".join([result.get('description', '') for result in search_data.get('web', {}).get('results', [])])
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
                max_tokens=700
            )
            st.markdown("---")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Groq Error: {e}")