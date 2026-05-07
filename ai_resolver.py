"""MIMS brand-to-generic resolver.

Loads data/mims_brand_generic_names.txt at startup.
Format: BRAND_NAME: generic_name (one per line)

Resolution order:
  1. MIMS exact match (instant, free)
  2. MIMS first-word match
  3. (AI fallback disabled per strict medical guidelines)
"""
import os
import re
from typing import Optional

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
    except FileNotFoundError:
        MIMS_LOAD_STATUS = f"file_not_found:{MIMS_DATA_PATH}"
        print(f"[MIMS] File not found: {MIMS_DATA_PATH}")
    except Exception as e:
        MIMS_LOAD_STATUS = f"error:{e}"
        print(f"[MIMS] Load error: {e}")

# Load on import
_load_mims()


def _gemini_resolve(brand_name: str, model) -> Optional[str]:
    """
    Ask Gemini to identify the generic drug name for a Philippine brand.
    Returns the generic name (lowercase) or None.
    NOTE: Currently bypassed to strictly prevent intuition/guessing.
    """
    return None


def ai_resolve_generic(brand_name: str, model=None):
    """
    Resolve brand name to generic.

    Priority:
      1. MIMS data file (exact match, instant)
      2. MIMS first-word match (e.g. "BIOGESIC 500MG" -> "BIOGESIC")
      3. AI fallback (Disabled per explicit constraint: do not guess if not in MIMS)
    """
    if not brand_name:
        return None

    # --- 1. Direct MIMS lookup (exact match, case-insensitive) ---
    key = brand_name.strip().upper()
    result = MIMS_BRAND_TO_GENERIC.get(key)
    if result:
        return result

    # --- 2. Try without common suffixes (e.g., "BIOGESIC 500MG" -> "BIOGESIC") ---
    words = key.split()
    if len(words) > 1:
        result = MIMS_BRAND_TO_GENERIC.get(words[0])
        if result:
            return result

    # --- 3. Gemini AI fallback (DISABLED) ---
    # User requested: "if no brand is located in the mims, model should not guess."
    return None

def get_mims_status():
    """Return MIMS loading status for health endpoint."""
    return {
        "mims_status": MIMS_LOAD_STATUS,
        "mims_entries": len(MIMS_BRAND_TO_GENERIC),
    }
