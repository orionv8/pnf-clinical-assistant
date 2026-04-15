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
# Drugs requiring special justification/clearance in PH hospitals
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

user_query = st.text_input("Generic, Brand, or Combination:", placeholder="e.g. 'Ceftriaxone' or 'Merronidazole + Azithromycin'")

if user_query:
    with st.spinner("Consulting Live DOH PNF Site..."):
        
        # --- DUAL-STAGE SEARCH (To fix blank monographs) ---
        headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY}
        
        # Search 1: General Listing
        q1 = f"site:pnf.doh.gov.ph {user_query}"
        # Search 2: Specific Monograph Details
        q2 = f"site:pnf.doh.gov.ph {user_query} indications contraindications dosage"
        
        web_context = ""
        try:
            for q in [q1, q2]:
                params = {"q": q, "count": 3}
                resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
                results = resp.json().get('web', {}).get('results', [])
                web_context += "\n".join([r.get('description', '') for r in results])
        except:
            web_context = "Web search unavailable."

        # --- AMS FLAG LOGIC ---
        is_restricted = any(drug in user_query.lower() for drug in AMS_RESTRICTED)

        # --- THE SMART PROMPT ---
        prompt = f"""
        USER QUERY: {user_query}
        WEB SEARCH CONTEXT: {web_context}

        STRICT INSTRUCTIONS:
        You are a Clinical Pharmacist. Your job is to summarize information from the Philippine National Formulary (PNF).
        DO NOT use pre-trained knowledge for dosages—STRICTLY use the WEB SEARCH CONTEXT.

        RULE 0: IF NOT FOUND
        If the web search contains no specific clinical data, output ONLY: "Drug not found in official PNF 8th Edition online portal."

        RULE 1: IF THE QUERY IS A SINGLE DRUG
        Follow this EXACT structure:
        
        Based strictly on the PNF 8th Edition online portal, here is the information for [Generic Name]:

        {'### ⚠️ AMS ALERT: RESTRICTED ANTIMICROBIAL' if is_restricted else ''}
        {'> This medicine is listed as a RESTRICTED antimicrobial in the PNF. Use requires institutional AMS clearance and specific clinical justification for PhilHealth reimbursement.' if is_restricted else ''}

        1. **Formulary Status**
        - **Classification:** [Classification from context]
        - **Available Forms & Strengths:** [List only what is in search results]

        2. **Clinical Monograph**
        - **Indications:** [Summarize indications from context]
        - **Contraindications:** [Summarize or write 'Not specified' if missing]
        - **Selected Dosage:** [List specific doses from context]

        *Note: [One key clinical pearl here]*
        """

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a clinical assistant. You provide structured, accurate medical summaries based ONLY on provided text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1000
            )
            st.markdown("---")
            st.write(response.choices[0].message.content)
            st.caption("🔍 Data sourced live from pnf.doh.gov.ph")
        except Exception as e:
            st.error(f"Groq Error: {e}")
