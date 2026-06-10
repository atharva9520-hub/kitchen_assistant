"""
profiles/cooking/model_router.py
---------------------------------
Cooking-specific model routing.

Overrides select_model(), check_frames(), print_routing_table() from core.

Routing:
  moondream — colour, liquid, doneness, general  (fast, 2-3s on M4)
  llava     — size, texture                      (better spatial detail, 8-10s)
"""

from core.vision_engine.model_router import (
    _get_available_models,
    query_model,
    MODEL_MOONDREAM,
    MODEL_LLAVA,
)
from prompts import detect_category


MOONDREAM_CATEGORIES = {"colour", "liquid", "doneness", "general"}
LLAVA_CATEGORIES     = {"size", "texture"}


def select_model(category: str) -> str | None:
    """Pick best available model for a cooking check category."""
    available = _get_available_models()

    if category in LLAVA_CATEGORIES:
        if MODEL_LLAVA     in available: return MODEL_LLAVA
        if MODEL_MOONDREAM in available:
            print(f"  [router] LLaVA not available — using moondream for {category}")
            return MODEL_MOONDREAM
    else:
        if MODEL_MOONDREAM in available: return MODEL_MOONDREAM
        if MODEL_LLAVA     in available:
            print(f"  [router] moondream not available — using LLaVA for {category}")
            return MODEL_LLAVA

    return None


def check_frames(
    frames_bytes: list[bytes],
    step:         dict,
    dish_name:    str,
) -> dict:
    """Auto-select model and evaluate a cooking step."""
    category = detect_category(step)
    model    = select_model(category)

    if model is None:
        return {
            "done":     False,
            "feedback": "No vision model available. Run: ollama pull moondream",
            "model":    "none",
            "category": category,
        }

    return query_model(frames_bytes, step, dish_name, model, category)


def print_routing_table():
    available = _get_available_models()
    print("\n  Cooking vision routing:")
    for cat in sorted(MOONDREAM_CATEGORIES | LLAVA_CATEGORIES):
        model = select_model(cat)
        print(f"    {cat:12s} → {model or 'NOT AVAILABLE'}")
    print()
