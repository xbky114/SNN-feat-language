"""Prompt construction for fixed three-level datasets."""

def build_prompt(text: str, father: str | None = None) -> str:
    """Build a CLIP prompt for a class, optionally conditioned on its parent."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if father is None:
        return f"a photo of {text}"
    if not isinstance(father, str) or not father.strip():
        raise ValueError("father must be None or a non-empty string")
    return f"a photo of {text}, which is one kind of {father}"
