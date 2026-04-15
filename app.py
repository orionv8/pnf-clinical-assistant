import streamlit as st
import os
import requests
from groq import Groq

# 1. Page Configuration
st.set_page_config(page_title="PNF Clinical Assistant", page_icon="💊")

# 2. API Key Loading
GROQ_KEY = os.getenv("GROQ_API_KEY")
BRAVE_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

# --- CLINICAL DATA: DOH RESTRICTED ANTIMICROBIALS (AMS) ---
AMS_RESTRICTED = [
    "cefepime", "ertapenem", "meropenem", "vancomycin", 
    "amphotericin b", "voriconazole", "colistin", 
    "micafungin", "aztreonam", "linezolid", "imipenem", "tigecycline"
]

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

# 4. Clinical UI
st.title("🇵🇭 PNF Clinical Assistant")
st.markdown("---")

user_query = st.text_input("Generic, Brand, or Combination:", placeholder="e.g. 'Vancocin' or 'Meropenem'")

if user_query:
    with st.spinner("Searching DOH Portal & Clinical Guidelines..."):
        
        # --- AGGRESSIVE DUAL-SEARCH LOGIC ---
        headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY}
        
        # Query 1: Direct DOH Portal lookup
        q1 = f"site:pnf.doh.gov.ph {user_query} monograph"
        # Query 2: Broader search to catch Vancomycin/Restricted drug details (PhilHealth/RITM/DOH PDFs)
        q2 = f"Philippine National Formulary {user_query} indications dosage strengths"
        
        web_context = ""
        try:
            for q in [q1, q2]:
                params = {"q": q, "count": 4}
                resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
                results = resp.json().get('web', {}).get('results', [])
                for r in results:
                    web_context += f"\n---\nSource: {r.get('url')}\nContent: {r.get('description', '')}"
        except:
            web_context = "Web search currently unavailable."

        # --- AMS FLAG LOGIC ---
        is_restricted = any(drug in user_query.lower() for drug in AMS_RESTRICTED)

        # --- THE CLINICAL PROMPT ---
        prompt = f"""
        USER QUERY: {user_query}
        WEB SEARCH CONTEXT: {web_context}

        STRICT INSTRUCTIONS:
        You are a Clinical Pharmacist. Summarize the Philippine National Formulary (PNF) data.
        
        RULE 0: IF NO DATA AT ALL
        If there is zero clinical data in the context, say: "Drug not found in official PNF 8th Edition online portal."

        RULE 1: FORMATTING
        Use the following structure. Do NOT invent dosages.
        
        Based strictly on PNF references and DOH protocols, here is the information for {user_query}:

        {'### ⚠️ AMS ALERT: RESTRICTED ANTIMICROBIAL' if is_restricted else ''}
        {'> **Note:** This is a RESTRICTED antimicrobial. Usage typically requires institutional AMS committee clearance and specific clinical justification for PhilHealth reimbursement.' if is_restricted else ''}

        1. **Formulary Status**
        - **Classification:** [Extract classification]
        - **Available Forms & Strengths:** [List only what is in the search results]

        2. **Clinical Monograph**
        - **Indications:** [List indications found]
        - **Contraindications:** [List or 'Not specified']
        - **Selected Dosage:** [List specific doses found]

        *Note: [Include one clinical pearl, e.g., infusion rate for Vancomycin or monitor renal function]*
        """

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a clinical assistant that strictly follows the PNF and DOH guidelines provided in the context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1000
            )
            st.markdown("---")
            st.write(response.choices[0].message.content)
            st.caption("🔍 Data fetched live from pnf.doh.gov.ph and DOH Clinical Guidelines.")
        except Exception as e:
            st.error(f"Groq Error: {e}")
