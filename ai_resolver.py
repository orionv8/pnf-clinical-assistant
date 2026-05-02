"""AI-powered brand-to-generic drug name resolver using Gemini."""
import re


def ai_resolve_generic(brand_name, model):
    """
    Use Gemini to resolve a brand-name drug to its generic/INN name.

    Args:
        brand_name: The brand name to resolve (e.g. "Biogesic")
        model: The Vertex AI GenerativeModel instance (_GEMMA_MODEL)

    Returns:
        The generic name string (e.g. "paracetamol"), or None if unavailable.
    """
    if model is None:
        return None
    try:
        prompt = (
            f"What is the generic (INN) drug name for the Philippine brand '{brand_name}'? "
            "Reply with ONLY the generic drug name in lowercase, nothing else. "
            "If you don't know, reply with just the word 'unknown'."
        )
        result = model.generate_content(prompt).text.strip().lower()
        # Clean: remove quotes, periods, extra punctuation
        result = re.sub(r'["\.,;!?]', '', result).strip()
        # Must be a plausible single drug name (1-3 words, under 40 chars)
        if result and result != "unknown" and 2 < len(result) < 40 and len(result.split()) <= 3:
            return result
    except Exception:
        pass
    return None
