import streamlit as st
import os
import requests
import json
import re
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader

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
        st.success("✅ Fast-Indexed PNF Engine Connected")

if not (GROQ_KEY and BRAVE_KEY):
    st.info("Awaiting API Keys to initialize...")
    st.stop()

# 3. Initialize Groq
groq_client = Groq(api_key=GROQ_KEY)

# 4. The Auto-Indexer Engine
@st.cache_resource
def load_or_build_index():
    index_file = "data/pnf_index.json"
    data_dir = "data"
    
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    all_pages = []
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.lower().endswith(".pdf"):
                try:
                    loader = PyPDFLoader(os.path.join(data_dir, filename))
                    docs = loader.load()
                    for doc in docs:
                        all_pages.append({"text": doc.page_content, "source": filename})
                except Exception as e:
                    pass
                    
        if all_pages:
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(all_pages, f)
                
    return all_pages

all_pnf_pages = load_or_build_index()

# 5. Clinical UI
st.title("🇵🇭 PNF Clinical Assistant")
st.markdown("---")

user_query = st.text_input("Enter Drug(s) or Ask a Clinical Question:", placeholder="e.g. 'Furosemide', 'Biogesic', or 'Can I combine Metronidazole and Azithromycin?'")

if user_query:
    with st.spinner("Scanning PNF Index & Guidelines..."):
        
        clean_query = user_query.lower().strip()
        is_restricted = any(drug in clean_query for drug in AMS_RESTRICTED)
        
        # Detect if it's a simple single drug lookup or a complex question/combo
        complex_triggers = ["+", "and", "&", "vs", ",", "interaction", "what", "how", "why", "can", "use of", "dose of", "safe"]
        is_complex = any(x in clean_query for x in complex_triggers)

        # --- SMART SCORED INDEX SEARCH (EML Priority) ---
        relevant_text = ""
        
        if all_pnf_pages:
            # 1. Strip punctuation from the query
            text_no_punct = re.sub(r'[^\w\s]', ' ', clean_query)
            raw_words = text_no_punct.split()
            
            # 2. Filter out conversational words so only clinical keywords remain
            stop_words = {"what", "is", "the", "use", "of", "can", "i", "combine", "and", "vs", "or", "with", "how", "much", "dose", "dosage", "tell", "me", "about", "are", "interactions", "between", "for", "a", "an", "to", "in", "on", "does", "have", "it", "safe", "will"}
            search_terms = [w for w in raw_words if w not in stop_words and len(w) > 2]
            
            # Fallback just in case they only typed stop words somehow
            if not search_terms:
                search_terms = [w for w in raw_words if len(w) > 2]
            
            scored_pages = []
            for p in all_pnf_pages:
                content_lower = p["text"].lower()
                term_count = sum(content_lower.count(term) for term in search_terms)
                
                if term_count > 0:
                    boost = 1
                    # Rule 1: Prioritize EML for formulary baseline
                    if "eml" in p["source"].lower():
                        boost += 3
                    # Rule 2: Prioritize monographs and interaction tables in the Manual
                    if any(x in content_lower for x in ["indication", "dosage", "contraindication", "interaction"]):
                        boost += 2
                        
                    # Prepend the source file to the text so the AI knows where it came from
                    tagged_text = f"[SOURCE: {p['source']}]\n{p['text']}"
                    scored_pages.append({"text": tagged_text, "score": term_count * boost})
            
            scored_pages.sort(key=lambda x: x["score"], reverse=True)
            best_pages = [page["text"] for page in scored_pages]
            
            # 20,000 character limit for deep reading
            relevant_text = "\n...\n".join(best_pages)[:20000]
            if not relevant_text:
                relevant_text = "[No data found in local Index]"
        else:
            relevant_text = "[Local Index missing or empty]"

        # --- BRAVE SEARCH (BRAND TRANSLATION ONLY) ---
        headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY}
        # Safely rebuild the search term without punctuation
        brand_search_query = " ".join(raw_words)
        search_term = f"{brand_search_query} generic name medicine philippines"
        params = {"q": search_term, "count": 2}
        
        web_context = ""
        try:
            resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            results = resp.json().get('web', {}).get('results', [])
            for r in results:
                web_context += f"Content: {r.get('description', '')}\n"
        except:
            web_context = "[Web search unavailable]"

        # --- DYNAMIC PROMPT INSTRUCTIONS ---
        if is_complex:
            template_instruction = """
            This is a COMPLEX QUERY (Combination, Interaction, or General Question).
            Do not use the standard single-drug template. Instead, provide a highly professional, comprehensive clinical answer structured with clear Markdown headings (e.g., Clinical Context, Interactions, Precautions). 
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
        
        DATA SOURCE 1 - LOCAL PDF INDEX: 
        {relevant_text}
        
        DATA SOURCE 2 - WEB SEARCH CONTEXT (FOR BRAND NAMES ONLY):
        {web_context}

        STRICT ARCHITECTURAL RULES:
        1. If the USER QUERY contains a BRAND NAME, look at DATA SOURCE 2 exclusively to find the generic equivalent. 
        2. DO NOT extract any dosages, indications, or clinical data from DATA SOURCE 2.
        3. ALL clinical data, interactions, and formulary statuses MUST be extracted from DATA SOURCE 1.
        4. When answering, prioritize data tagged with [SOURCE: PNF-EML_8th.pdf] for available strengths and classifications. Prioritize [SOURCE: PNF-Manual-for-Primary-Healthcare_8th.pdf] for dosages, indications, and interactions.
        5. If DATA SOURCE 1 contains zero information about the queried generic drug, output ONLY: "Drug or clinical information not found in official PNF indexed references."
        
        {template_instruction}
        """

        # --- GROQ AI RESPONSE ---
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a clinical AI. You strictly separate web data (brand translation only) from local index data (clinical facts). You never invent dosages."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1000
            )
            st.markdown("---")
            st.write(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Groq Error: {e}")
