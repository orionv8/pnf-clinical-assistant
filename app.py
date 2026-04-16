import streamlit as st
import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader

# 1. Page Configuration
st.set_page_config(page_title="PNF Clinical Assistant", page_icon="💊")

# 2. API Key Loading (Brave removed, Groq kept)
GROQ_KEY = os.getenv("GROQ_API_KEY")

# --- CLINICAL DATA: DOH RESTRICTED ANTIMICROBIALS (AMS) ---
AMS_RESTRICTED = [
    "cefepime", "ertapenem", "meropenem", "vancomycin", 
    "amphotericin b", "voriconazole", "colistin", 
    "micafungin", "aztreonam", "linezolid", "imipenem", "tigecycline"
]

with st.sidebar:
    st.header("⚙️ System Status")
    if not GROQ_KEY:
        st.error("Groq Key missing in Railway Variables")
        GROQ_KEY = st.text_input("Manual Groq API Key", value=GROQ_KEY if GROQ_KEY else "", type="password")
    else:
        st.success("✅ PNF Engine Connected")

if not GROQ_KEY:
    st.info("Awaiting API Key to initialize...")
    st.stop()

# 3. Initialize Groq
groq_client = Groq(api_key=GROQ_KEY)

# --- PDF DATA LOADER (Loads ALL PDFs in the data folder) ---
@st.cache_resource
def load_all_pdfs():
    all_pages = []
    data_dir = "data"
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.lower().endswith(".pdf"):
                try:
                    loader = PyPDFLoader(os.path.join(data_dir, filename))
                    all_pages.extend(loader.load())
                except Exception as e:
                    pass
    return all_pages

all_pnf_pages = load_all_pdfs()

# 4. Clinical UI
st.title("🇵🇭 PNF Clinical Assistant")
st.markdown("---")

user_query = st.text_input("Enter Drug(s):", placeholder="e.g. 'Ceftriaxone' or 'Amoxicillin + Clavulanic Acid'")

if user_query:
    with st.spinner("Scanning Local PDFs..."):
        
        # Clean the query so the search engine doesn't get confused by questions
        clean_query = user_query.lower().replace("what is", "").replace("the dose of", "").replace("for", "").replace("?", "").strip()
        
        # Detect if it is a combination or restricted drug
        is_combo = any(x in clean_query for x in ["+", "and", "&", "vs", ","])
        is_restricted = any(drug in clean_query for drug in AMS_RESTRICTED)

        # --- SMART TARGETED PDF SEARCH ---
        relevant_text = ""
        matched_count = 0
        
        if 'all_pnf_pages' in locals() and all_pnf_pages:
            search_terms = [t.strip().lower() for t in clean_query.replace("+", ",").replace("and", ",").replace("&", ",").split(",") if len(t.strip()) > 2]
            
            matched_pages = []
            for p in all_pnf_pages:
                content_lower = p.page_content.lower()
                if any(term in content_lower for term in search_terms):
                    matched_pages.append(p.page_content)
            
            matched_count = len(matched_pages)
            
            best_pages = []
            for page in matched_pages:
                if "indication" in page.lower() or "dosage" in page.lower() or "contraindication" in page.lower():
                    best_pages.insert(0, page) 
                else:
                    best_pages.append(page)
                    
            relevant_text = "\n...\n".join(best_pages)[:8000]
            if not relevant_text:
                relevant_text = "Drug not found in local PDFs."
        else:
            relevant_text = "Local PNF Manuals completely missing. Please upload PDFs to the 'data' folder."


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
        LOCAL PDF CONTEXT: {relevant_text}

        STRICT INSTRUCTIONS:
        You are a Clinical Pharmacist. 
        If the LOCAL PDF CONTEXT contains no clinical data, output ONLY: "Drug not found in official PNF portal."
        DO NOT invent dosages. STRICTLY use the LOCAL PDF CONTEXT.
        
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
                st.caption(f"**PDF Pages Matched:** `{matched_count}`")
                st.caption("**Raw Text extracted from PDFs:**")
                st.write(relevant_text if relevant_text else "No text extracted from local PDFs.")
                
        except Exception as e:
            st.error(f"Groq Error: {e}")
