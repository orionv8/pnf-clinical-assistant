"""MIMS brand-to-generic resolver.

Loads data/mims_brand_generic_names.txt at startup.
Format: BRAND_NAME: generic_name (one per line)
"""
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MIMS_DATA_PATH = os.path.join(BASE_DIR, "data", "mims_brand_generic_names.txt")

# Brand (uppercase) -> generic (lowercase) mapping
MIMS_BRAND_TO_GENERIC = {}
MIMS_LOAD_STATUS = "not_loaded"

def _load_mims():
    global MIMS_BRAND_TO_GENERIC, MIMS_LOAD_STATUS
    if MIMS_BRAND_TO_GENERIC:
        return
    try:
        with open(MIMS_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    brand, generic = line.split(":", 1)
                    brand = brand.strip().upper()
                    generic = generic.strip().lower()
                    if brand and generic:
                        MIMS_BRAND_TO_GENERIC[brand] = generic
        MIMS_LOAD_STATUS = f"loaded_{len(MIMS_BRAND_TO_GENERIC)}_entries"
        print(f"[MIMS] Loaded {len(MIMS_BRAND_TO_GENERIC)} brand-generic mappings")
        # Print first 5 for diagnostics
        for i, (k, v) in enumerate(MIMS_BRAND_TO_GENERIC.items()):
            if i >= 5:
                break
            print(f"  {k} -> {v}")
    except FileNotFoundError:
        MIMS_LOAD_STATUS = f"file_not_found:{MIMS_DATA_PATH}"
        print(f"[MIMS] File not found: {MIMS_DATA_PATH}")
    except Exception as e:
        MIMS_LOAD_STATUS = f"error:{e}"
        print(f"[MIMS] Load error: {e}")

# Load on import
_load_mims()


def ai_resolve_generic(brand_name: str, model=None):
    """
    Resolve brand name to generic using MIMS data ONLY.
    No AI/Gemma fallback — all answers must come from the data file.
    """
    if not brand_name:
        return None

    # Direct lookup (exact match, case-insensitive)
    key = brand_name.strip().upper()
    result = MIMS_BRAND_TO_GENERIC.get(key)
    if result:
        return result

    # Try without common suffixes (e.g., "BIOGESIC 500MG" -> "BIOGESIC")
    words = key.split()
    if len(words) > 1:
        result = MIMS_BRAND_TO_GENERIC.get(words[0])
        if result:
            return result

    return None


def get_mims_status():
    """Return MIMS loading status for health endpoint."""
    return {
        "mims_status": MIMS_LOAD_STATUS,
        "mims_entries": len(MIMS_BRAND_TO_GENERIC),
    }
