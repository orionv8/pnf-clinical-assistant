def ai_resolve_generic(brand_name: str, model) -> str | None:
    """
    Attempts to resolve a brand name to a generic name using the AI model as a last resort.
    Strictly constrained to prevent hallucinations on unknown local brands.
    """
    if model is None:
        return None

    # Strong few-shot prompt to enforce zero-guessing
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
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

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