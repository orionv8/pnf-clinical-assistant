"""Brave Search brand-to-generic resolver with PNF cross-reference.

Instead of parsing search result titles with regex, this scans
Brave result text for any drug name that exists in the PNF index.
Guarantees the result is always a valid PNF drug.
"""
import re
import os


def brave_resolve_generic(brand_name, pnf_data):
    """
    Use Brave Search to find the generic/INN name for a brand drug.
    Cross-references results against pnf_data to ensure validity.

    Args:
        brand_name: The brand name to look up (e.g. "Biogesic")
        pnf_data: The loaded PNF index list (each entry has 'drug' key)

    Returns:
        Generic drug name string (e.g. "paracetamol"), or None.
    """
    try:
        import requests
    except ImportError:
        return None

    brave_key = os.getenv("BRAVE_SEARCH")
    if not brave_key:
        return None

    try:
        headers = {"X-Subscription-Token": brave_key}
        url = (
            "https://api.search.brave.com/res/v1/web/search"
            f"?q={brand_name}+generic+name+drug+Philippines"
        )
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            results = data.get("web", {}).get("results", [])
            brand_lower = brand_name.lower().strip()

            # Build set of PNF drug names for fast lookup
            pnf_drugs = {
                e.get("drug", "").lower().strip()
                for e in pnf_data
                if e.get("drug", "").strip()
            }

            # Scan Brave results for any PNF drug name
            for result in results[:5]:
                combined = (
                    result.get("title", "") + " " +
                    result.get("description", "")
                ).lower()
                for drug in pnf_drugs:
                    if drug and drug != brand_lower and drug in combined:
                        return drug
    except Exception:
        pass
    return None
