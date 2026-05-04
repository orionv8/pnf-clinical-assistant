import re
import os

# Path to the MIMS data file
MIMS_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "mims_brand_generic_names.txt")

MIMS_BRAND_TO_GENERIC = {}

def _load_mims_mappings():
    """Loads MIMS brand-to-generic mappings from the data file."""
    global MIMS_BRAND_TO_GENERIC
    if MIMS_BRAND_TO_GENERIC: # Already loaded
        return

    try:
        with open(MIMS_DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    brand, generic = line.split(':', 1)
                    MIMS_BRAND_TO_GENERIC[brand.strip().upper()] = generic.strip().lower()
        print(f"Loaded {len(MIMS_BRAND_TO_GENERIC)} MIMS brand-generic mappings.")
    except FileNotFoundError:
        print(f"MIMS data file not found at {MIMS_DATA_PATH}. AI resolver will operate without MIMS data.")
    except Exception as e:
        print(f"Error loading MIMS data: {e}")

# Load mappings when the module is imported
_load_mims_mappings()

def ai_resolve_generic(brand_name: str, model) -> str | None:
    """
    Attempts to resolve a brand name to a generic name using MIMS data first,
    then falling back to the AI model as a last resort.
    """
    # 1. Check MIMS data first
    mims_generic = MIMS_BRAND_TO_GENERIC.get(brand_name.upper())
    if mims_generic:
        return mims_generic

    # 2. If not found in MIMS, use AI model (existing logic)
    if model is None:
        return None

    system_prompt = """You are a strict data-extraction clinical assistant. Your ONLY task is to map Philippine brand-name drugs to their generic (INN) names.

CRITICAL RULES:
1. Reply with ONLY the generic drug name in lowercase.
2. YOU MUST NEVER GUESS.
3. If the brand is obscure, not explicitly in your knowledge, or you are even slightly uncertain, reply with ONLY the word 'unknown'.

EXAMPLES:
Brand: 'biogesic'
paracetamol

Brand: 'madeupbrand_xyz'
unknown

Brand: 'solmux'
carbocisteine

Brand: 'obscure_local_med'
unknown"""

    user_prompt = f"Brand: '{brand_name}'"
    full_prompt = f"{system_prompt}\\n\\n{user_prompt}"

    try:
        # Temperature 0.0 is critical to stop hallucinations
        response = model.generate_content(
            full_prompt,
            generation_config={"temperature": 0.0}
        )

        result = response.text.strip().lower()

        # If the model follows instructions and admits it doesn't know, return None
        if result == 'unknown' or result == '':
            return None

        return result

    except Exception as e:
        print(f"AI Resolver error: {e}")
        return None
