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

# --- SOURCE NAME MAPPER ---
SOURCE_MAP = {
    "PNF-EML_8th.pdf": "Philippine National Formulary - Essential Medicines List (8th Ed.)",
    "PNF-Manual-for-Primary-Healthcare_8th.pdf": "Philippine National Formulary - Primary Healthcare Manual (8th Ed.)"
}

with st.sidebar:
    st.header("⚙️ System Status")
    if not GROQ_KEY or not BRAVE_KEY:
        st.error("Keys missing in Railway Variables")
        GROQ_KEY = st.text_input("Manual Groq API Key", value=GROQ_KEY if GROQ_KEY else "", type="password")
        BRAVE_KEY = st.text_input("Manual Brave API Key", value=BRAVE_KEY if BRAVE_KEY else "", type="password")
    else:
        st.success("✅ PNF Clinical Assistant Online")

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
                
            readable_source = SOURCE_MAP.get(p['source'], p['source'])
            tagged_text = f"[SOURCE: {readable_source}]\n{p['text']}"
            scored_pages.append({"text": tagged_text, "score": term_count * boost})
            
    scored_pages.sort(key=lambda x: x["score"], reverse=True)
    return scored_pages, search_terms


# 5. Clinical UI
st.title("🇵🇭 PNF Clinical Assistant")
st.markdown("---")

user_query = st.text_input("Enter Drug(s) or Ask a Question:", placeholder="e.g. 'Furosemide', 'Biogesic', or 'Metronidazole and Azithromycin?'")

if user_query:
    with st.spinner("Scanning PNF Index & Medical Web..."):
        
        clean_query = user_query.lower().strip()
        is_restricted = any(drug in clean_query for drug in AMS_RESTRICTED)
        
        complex_triggers = ["+", "and", "&", "vs", ",", "interaction", "what", "how", "why", "can", "use of", "dose of", "safe"]
        is_complex = any(x in clean_query for x in complex_triggers)

        # --- PHASE 1: LOCAL GATEKEEPER ---
        if not all_pnf_pages:
            st.error("Error: pnf_index.json not found in data folder.")
            st.stop()

        scored_pages, active_search_terms = search_local_index(clean_query, all_pnf_pages)
        
        # Format Local Text (Capped at 3500 chars to save tokens)
        best_pages = [page["text"] for page in scored_pages]
        relevant_text = "\n...\n".join(best_pages)[:3500] 
        if not relevant_text:
            relevant_text = "[No data found in local PNF Index]"

        # --- PHASE 2: THE WEB FALLBACK (Brand Translation OR Missing Clinical Data) ---
        web_context = ""
        
        # We search the web if there are 0 local hits (brand name) OR if it's a complex question (interactions)
        if len(scored_pages) == 0 or is_complex:
            headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY}
            search_query_string = " ".join(active_search_terms)
            
            # If 0 hits, it's likely a brand name. If complex, look for interactions.
            if len(scored_pages) == 0:
                params = {"q": f"{search_query_string} generic name medicine philippines", "count": 2}
            else:
                params = {"q": f"{search_query_string} drug interactions clinical guidelines", "count": 3}
            
            try:
                resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
                results = resp.json().get('web', {}).get('results', [])
                
                for r in results:
                    web_context += f"[SOURCE: {r.get('url', 'Web Article')}]\nContent: {r.get('description', '')}\n\n"
                    
                # Strict Web Cap: 1500 chars to prevent Token Limit 413 Error
                web_context = web_context[:1500]
                
            except Exception as e:
                web_context = "[Web search unavailable]"
        else:
            # Leave this completely empty so the AI doesn't cite "Search Bypassed" in its references
            web_context = "" 


        # --- DYNAMIC PROMPT INSTRUCTIONS ---
        if is_complex:
            template_instruction = """
            This is a COMPLEX QUERY (Combination, Interaction, or General Question).
            Provide a highly professional, comprehensive clinical answer structured with clear Markdown headings (e.g., Clinical Context, Interactions, Precautions). 
            """
        else:
            template_instruction = f"""
            This is a SINGLE DRUG query. You MUST use this EXACT structure:
            
            Based on the references, here is the information for [Insert Generic Name Here] (if the user queried a brand name, add the brand name in parentheses here, e.g. "Paracetamol (Biogesic)"):

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
        
        DATA SOURCE 1 - LOCAL PNF INDEX (Primary Authority): 
        {relevant_text}
        
        DATA SOURCE 2 - WEB MEDICAL CONTEXT (Fallback Data):
        {web_context}
        
        STRICT ARCHITECTURAL RULES:
        1. Prioritize DATA SOURCE 1 for all base facts. Use DATA SOURCE 2 to fill in any missing gaps (especially for drug interactions or translating brand names).
        2. Never invent dosages. If neither source has the answer, state that the information is unavailable.
        3. AT THE VERY BOTTOM of your response, add a section called "### References". List all the specific sources you used (e.g., "Philippine National Formulary - Primary Healthcare Manual (8th Ed.)" and any specific Web URLs provided in the source brackets). DO NOT cite "Fallback Data" or internal system text as a reference.
        
        {template_instruction}
        """

        # --- GROQ AI RESPONSE ---
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a clinical AI. You format references clearly at the bottom of your output."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=850
            )
            st.markdown("---")
            st.write(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Groq Error: {e}")

# --- FOOTER & DISCLAIMER ---
st.markdown("---")
st.caption("ℹ️ **About PNF Clinical Assistant:** This tool sources its information directly from the Philippine National Formulary (PNF) Essential Medicines List and Primary Healthcare Manual, supplemented by web searches for brand-to-generic translations and complex interactions.")
st.caption("⚠️ **Disclaimer:** While we strive for accuracy and up-to-date information, this tool is for quick reference only and may occasionally miss details. Always verify dosages and critical protocols directly with the official [DOH PNF Website](https://pnf.doh.gov.ph/) or your institutional guidelines if in doubt.")
