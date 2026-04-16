import streamlit as st
import os
import requests
import json
import re
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
        st.success("✅ Sequential PNF Engine Connected")

if not (GROQ_KEY and BRAVE_KEY):
    st.info("Awaiting API Keys to initialize...")
    st.stop()

# 3. Initialize Groq
groq_client = Groq(api_key=GROQ_KEY)

# 4. Instant Static Index Loader
@st.cache_resource
def load_static_index():
    index_file = "data/pnf_index.json"
    if os.path.exists(index_file):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading index: {e}")
            return []
    return []

all_pnf_pages = load_static_index()

# --- HELPER FUNCTION: LOCAL SEARCH ENGINE ---
def search_local_index(query, index_data):
    text_no_punct = re.sub(r'[^\w\s]', ' ', query.lower().strip())
    raw_words = text_no_punct.split()
    
    stop_words = {"what", "is", "the", "use", "of", "can", "i", "combine", "and", "vs", "or", "with", "how", "much", "dose", "dosage", "tell", "me", "about", "are", "interactions", "between", "for", "a", "an", "to", "in", "on", "does", "have", "it", "safe", "will"}
    search_terms = [w for w in raw_words if w not in stop_words and len(w) > 2]
    
    if not search_terms:
        search_terms = [w for w in raw_words if len(w) > 2]
        
    scored_pages = []
    for p in index_data:
        content_lower = p["text"].lower()
        term_count = sum(content_lower.count(term) for term in search_terms)
        
        if term_count > 0:
            boost = 1
            if "eml" in p["source"].lower():
                boost += 3
            if any(x in content_lower for x in ["indication", "dosage", "contraindication", "interaction"]):
                boost += 2
                
            tagged_text = f"[SOURCE: {p['source']}]\n{p['text']}"
            scored_pages.append({"text": tagged_text, "score": term_count * boost})
            
    scored_pages.sort(key=lambda x: x["score"], reverse=True)
    return scored_pages, search_terms


# 5. Clinical UI
st.title("🇵🇭 PNF Clinical Assistant")
st.markdown("---")

user_query = st.text_input("Enter Drug(s) or Ask a Question:", placeholder="e.g. 'Furosemide', 'Biogesic', or 'Can I combine Metronidazole and Azithromycin?'")

if user_query:
    with st.spinner("Scanning Local PNF Index..."):
        
        clean_query = user_query.lower().strip()
        is_restricted = any(drug in clean_query for drug in AMS_RESTRICTED)
        
        complex_triggers = ["+", "and", "&", "vs", ",", "interaction", "what", "how", "why", "can", "use of", "dose of", "safe"]
        is_complex = any(x in clean_query for x in complex_triggers)

        # --- PHASE 1: LOCAL GATEKEEPER ---
        if not all_pnf_pages:
            st.error("Error: pnf_index.json not found in data folder.")
            st.stop()

        scored_pages, active_search_terms = search_local_index(clean_query, all_pnf_pages)
        
        web_context = "[Web Search Bypassed - Local Matches Found]"
        
        # --- PHASE 2 & 3: THE WEB RESCUE (BRAND NAME TRANSLATION) ---
        if len(scored_pages) == 0:
            st.info("No local matches found. Checking web for brand name equivalent...")
            
            headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY}
            search_query_string = " ".join(active_search_terms)
            params = {"q": f"{search_query_string} generic name medicine philippines", "count": 2}
            
            try:
                resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
                results = resp.json().get('web', {}).get('results', [])
                raw_web_text = " ".join([r.get('description', '') for r in results])[:800]
                
                # Micro-AI Call: Extract Generic Name ONLY
                trans_prompt = f"Read this web search snippet and reply with ONLY the generic drug name. Do not write anything else. Snippet: {raw_web_text}"
                trans_response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": trans_prompt}],
                    temperature=0.0,
                    max_tokens=20
                )
                
                generic_name = trans_response.choices[0].message.content.strip().lower()
                web_context = f"Identified Brand Name. Translating '{search_query_string}' to Generic: {generic_name.title()}"
                
                # RE-ROUTE: Search the Local Index again using the new Generic Name!
                scored_pages, _ = search_local_index(generic_name, all_pnf_pages)
                
            except Exception as e:
                web_context = "[Web search failed or no generic found]"

        # --- PREPARE FINAL CONTEXT ---
        best_pages = [page["text"] for page in scored_pages]
        relevant_text = "\n...\n".join(best_pages)[:5000] # Kept at 5000 to protect your Token Limits
        
        if not relevant_text:
            relevant_text = "[No data found in local Index, even after brand translation]"

        # --- DYNAMIC PROMPT INSTRUCTIONS ---
        if is_complex:
            template_instruction = """
            This is a COMPLEX QUERY (Combination, Interaction, or General Question).
            Provide a highly professional, comprehensive clinical answer structured with clear Markdown headings (e.g., Clinical Context, Interactions, Precautions). 
            Base your entire answer ONLY on the LOCAL PDF INDEX provided.
            """
        else:
            template_instruction = f"""
            This is a SINGLE DRUG query. You MUST use this EXACT structure:
            
            Based strictly on the PNF references, here is the information for {clean_query.title()}:

            {'### ⚠️ AMS ALERT: RESTRICTED ANTIMICROBIAL' if is_restricted else ''}
            {'> **Note:** This medicine is a RESTRICTED antimicrobial. Usage requires institutional AMS clearance and specific justification.' if is_restricted else ''}

            1. **Formulary Status (Source: PNF-EML)**
            - **Classification:** [Classification]
            - **Available Forms & Strengths:** [List forms found in context]

            2. **Clinical Monograph (Source: PNF Primary Healthcare Manual)**
            - **Indications:** [List indications]
            - **Contraindications:** [List or write 'Not specified']
            - **Selected Dosage:** [List specific doses]
            - **Key Interactions:** [Briefly list major interactions if found]
            """

        prompt = f"""
        USER QUERY: {clean_query}
        
        WEB CONTEXT (Translation Status): {web_context}
        
        LOCAL PDF INDEX (Clinical Facts): 
        {relevant_text}
        
        STRICT ARCHITECTURAL RULES:
        1. ALL clinical data, interactions, and formulary statuses MUST be extracted from the LOCAL PDF INDEX.
        2. Prioritize data tagged with [SOURCE: PNF-EML] for strengths and classifications. Prioritize [SOURCE: PNF-Manual] for dosages and interactions.
        3. If the LOCAL PDF INDEX says "[No data found]", output ONLY: "Drug or clinical information not found in official PNF indexed references." Do not invent an answer.
        
        {template_instruction}
        """

        # --- GROQ AI RESPONSE ---
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a clinical AI. You never invent dosages and rely exclusively on the local index."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=850
            )
            st.markdown("---")
            
            # If the web rescue was used, add a small disclaimer at the top
            if "Translating" in web_context:
                st.info(f"🔄 **Brand Translation:** {web_context}")
                
            st.write(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Groq Error: {e}")
