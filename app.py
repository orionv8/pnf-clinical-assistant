import streamlit as st
import os
import requests
from groq import Groq

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

# 4. Clinical UI
st.title("🇵🇭 PNF Clinical Assistant")
st.markdown("---")

user_query = st.text_input("Generic, Brand, or Combination:", placeholder="e.g. 'Ceftriaxone' or 'Metronidazole + Azithromycin'")

if user_query:
    with st.spinner("Consulting Live DOH PNF Site..."):
        # --- BRAVE SEARCH (Targeted to DOH) ---
        headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY}
        # Force Brave to ONLY pull from the official pnf.doh.gov.ph portal
        params = {"q": f"site:pnf.doh.gov.ph {user_query}", "count": 5} 
        
        web_context = ""
        try:
            search_resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            search_data = search_resp.json()
            results = search_data.get('web', {}).get('results', [])
            web_context = "\n".join([f"Source: {r.get('url')}\nContent: {r.get('description', '')}" for r in results])
        except:
            web_context = "Web search unavailable."

        # --- THE "WEB-ONLY" SMART PROMPT ---
        prompt = f"""
        USER QUERY: {user_query}
        WEB SEARCH CONTEXT: {web_context}

        STRICT INSTRUCTIONS:
        You are a Clinical Pharmacist. Your job is to summarize information from the Philippine National Formulary (PNF).
        DO NOT use your pre-trained knowledge for dosages—STRICTLY use the WEB SEARCH CONTEXT provided.

        RULE 0: IF NOT FOUND
        If the web search contains no specific clinical data for this drug from pnf.doh.gov.ph, output:
        "Drug not found in official PNF 8th Edition online portal."

        RULE 1: IF THE QUERY IS A SINGLE DRUG
        Use this EXACT structure:
        
        Based strictly on the PNF 8th Edition online portal, here is the information for [Generic Name]:

        1. Formulary Status
        - **Classification:** [Classification from context]
        - **Available Forms & Strengths:** [List only what is in the search results]

        2. Clinical Monograph
        - **Indications:** [List]
        - **Contraindications:** [List]
        - **Selected Dosage:** [List specific doses]

        Note: [Include one key clinical pearl here]

        RULE 2: IF THE QUERY IS A COMBINATION (e.g., A + B)
        Use this structure:
        
        Based on PNF protocols, here is the clinical context for combining [Drug A] and [Drug B]:
        
        1. Clinical Context & Rationale
        - [Explain the combined protocol or syndromic management]
        
        2. Protocol Regimen
        - **[Drug A]:** [Dose]
        - **[Drug B]:** [Dose]
        
        3. Key PNF Precautions
        - [List interactions or warnings]
        """

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a direct Clinical AI. You never echo the prompt. You use only provided web data for medical values."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=850
            )
            st.markdown("---")
            st.write(response.choices[0].message.content)
            st.caption("🔍 Source: Live search of pnf.doh.gov.ph")
        except Exception as e:
            st.error(f"Groq Error: {e}")
