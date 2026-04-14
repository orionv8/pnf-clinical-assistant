import streamlit as st
import os
import requests
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader

# 1. Page Configuration
st.set_page_config(page_title="PNF Clinical Assistant", page_icon="💊")

# 2. API Key Loading 
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

# 4. PNF Data Loading
@st.cache_resource
def load_pnf_context():
    path = os.path.join("data", "PNF-Manual-for-Primary-Healthcare_8th.pdf")
    if os.path.exists(path):
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
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
    with st.spinner("Analyzing PNF References..."):
        is_combo = any(x in user_query.lower() for x in ["+", "and", "&", "interaction", "with", "vs"])
        
        # BRAVE SEARCH
        headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY}
        params = {"q": f"Philippine National Formulary PNF 8th edition {user_query} protocol", "count": 3}
        
        try:
            search_resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            search_data = search_resp.json()
            results = search_data.get('web', {}).get('results', [])
            web_context = "\n".join([r.get('description', '') for r in results])
        except:
            web_context = "Web search unavailable."

        # STRICT TEMPLATED PROMPT
        prompt = f"""
        USER QUERY: {user_query}
        LOCAL PNF DATA: {pnf_context[:4000]}
        WEB SEARCH: {web_context}

        STRICT INSTRUCTIONS:
        You are a Clinical Pharmacist. Follow these formatting rules EXACTLY based on the query type. Do NOT echo the prompt or web search data in your response.

        RULE 0: IF NOT FOUND OR IRRELEVANT
        If the query is not a recognized drug, or if there is no clinical data found in the LOCAL PNF DATA or WEB SEARCH, output EXACTLY this phrase and nothing else:
        "Drug not found in PNF 8th Edition."

        RULE 1: IF THE QUERY IS A SINGLE DRUG
        Use this EXACT template and structure. Do not change the headings.
        
        Based strictly on the PNF 8th Edition and the PNF Manual for Primary Healthcare, here is the information for [Generic Name] (listed as [Full PNF Listing Name]):

        1. Formulary Status (PNF 8th Edition, 2017)
        - **Classification:** [Classification]
        - **Available Forms & Strengths:**
          - [List forms and strengths]

        2. Clinical Monograph (PNF Manual for Primary Healthcare)
        - **Indications:**
          - [List indications]
        - **Contraindications:**
          - [List contraindications]
        - **Common Adverse Reactions:** [List]
        - **Selected Dosage:**
          - [List specific doses]

        Note: [Include one key clinical pearl or PNF usage note here]


        RULE 2: IF THE QUERY IS A COMBINATION OR INTERACTION (e.g., A + B)
        Provide strong clinical context without giving two completely separate monographs. Use this EXACT structure:
        
        Based strictly on the PNF 8th Edition and primary healthcare protocols, here is the clinical context for combining [Drug A] and [Drug B]:
        
        1. Clinical Context & Rationale
        - [Explain WHY they are used together. e.g., Syndromic management of STIs, covering specific organisms, or overlapping mechanisms.]
        
        2. Protocol Regimen & Selected Dosage
        - **[Drug A]:** [Dose in this specific protocol]
        - **[Drug B]:** [Dose in this specific protocol]
        
        3. Key PNF Precautions & Interactions
        - [Detail specific drug-drug interactions, overlapping toxicities, or warnings like avoiding alcohol or QTc prolongation.]
        """



        

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a highly structured clinical AI. You fill in the provided templates exactly as requested without adding conversational filler."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=850
            )
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Groq Error: {e}")
