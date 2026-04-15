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

user_query = st.text_input("Enter Drug(s):", placeholder="e.g. 'Ceftriaxone' or 'Amoxicillin + Clavulanic Acid'")

if user_query:
    with st.spinner("Searching DOH Portal..."):
        
        # Clean the query so the search engine doesn't get confused by questions
        clean_query = user_query.lower().replace("what is", "").replace("the dose of", "").replace("for", "").replace("?", "").strip()
        
        # Detect if it is a combination or restricted drug
        is_combo = any(x in clean_query for x in ["+", "and", "&", "vs", ","])
        is_restricted = any(drug in clean_query for drug in AMS_RESTRICTED)

        # --- BRAVE SEARCH ---
        headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY}
        
        # Smart Search: If restricted, broaden the search to catch PhilHealth PDFs. Otherwise, lock to PNF.
        if is_restricted:
            search_term = f"Philippine National Formulary {clean_query} monograph dosage"
        else:
            search_term = f"site:pnf.doh.gov.ph {clean_query}"
            
        params = {"q": search_term, "count": 5}
        
        web_context = ""
        try:
            resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            results = resp.json().get('web', {}).get('results', [])
            for r in results:
                web_context += f"\nSource: {r.get('title')}\nContent: {r.get('description', '')}\n"
        except:
            web_context = "Web search unavailable."

        # --- DYNAMIC PROMPT INSTRUCTIONS ---
        if is_combo:
            template_instruction = """
            This is a COMBINATION/MULTIPLE DRUG query. You MUST use this EXACT structure:
            
            Based strictly on PNF protocols, here is the clinical context for combining these medicines:
            
            1. **Clinical Context & Rationale**
            - [Explain why they are used together or potential interactions]
            
            2. **Protocol Regimen & Selected Dosages**
            - [List the specific doses for each drug found in the context]
            
            3. **Key PNF Precautions**
            - [List specific warnings or overlapping toxicities]
            """
        else:
            template_instruction = f"""
            This is a SINGLE DRUG query. You MUST use this EXACT structure:
            
            Based strictly on the PNF online portal, here is the information for {clean_query.title()}:

            {'### ⚠️ AMS ALERT: RESTRICTED ANTIMICROBIAL' if is_restricted else ''}
            {'> **Note:** This medicine is a RESTRICTED antimicrobial. Usage requires institutional AMS clearance and specific justification.' if is_restricted else ''}

            1. **Formulary Status**
            - **Classification:** [Classification]
            - **Available Forms & Strengths:** [List forms found in context]

            2. **Clinical Monograph**
            - **Indications:** [List indications]
            - **Contraindications:** [List or write 'Not specified']
            - **Selected Dosage:** [List specific doses]
            """

        prompt = f"""
        USER QUERY: {clean_query}
        WEB SEARCH CONTEXT: {web_context}

        STRICT INSTRUCTIONS:
        You are a Clinical Pharmacist. 
        If the WEB SEARCH CONTEXT contains no clinical data, output ONLY: "Drug not found in official PNF portal."
        DO NOT invent dosages. STRICTLY use the WEB SEARCH CONTEXT.
        
        {template_instruction}
        """

        # --- GROQ AI RESPONSE ---
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a highly accurate clinical AI. You strictly follow the formatting template provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=850
            )
            st.markdown("---")
            st.write(response.choices[0].message.content)
            
            # --- THE DIAGNOSTIC VIEWER ---
            with st.expander("👀 See what the bot searched (Debug Viewer)"):
                st.caption(f"**Exact text sent to Brave Search:** `{search_term}`")
                st.caption("**Raw Data retrieved from DOH:**")
                st.write(web_context if web_context else "No data returned from search engine.")
                
        except Exception as e:
            st.error(f"Groq Error: {e}")
