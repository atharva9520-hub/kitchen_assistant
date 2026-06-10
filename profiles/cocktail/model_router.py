"""
model_router.py  (cocktail edition)
-------------------------------------
Routes each vision check to the right model.
Now also handles multi-frame inputs for motion steps (shake/stir/muddle).

Routing:
  moondream — ingredient ID, pour, garnish, presentation, strain (fast)
  llava     — motion steps (shake/stir/muddle) — better at spatial sequences
              falls back to moondream if llava unavailable
"""

import base64
import ollama
from prompts import build_prompt, parse_response, detect_category


MODEL_MOONDREAM = "moondream"
MODEL_LLAVA     = "llava"

MOONDREAM_CATEGORIES = {"ingredient", "pour", "garnish", "presentation", "strain", "general"}
LLAVA_CATEGORIES     = {"shake", "stir", "muddle"}

_available_models: set[str] | None = None


def _get_available_models() -> set[str]:
    global _available_models
    if _available_models is not None:
        return _available_models
    try:
        models = ollama.list()
        names  = {m["model"].split(":")[0] for m in models.get("models", [])}
        _available_models = names
        print(f"  [router] Available models: {names}")
    except Exception as e:
        print(f"  [router] Could not query Ollama: {e}")
        _available_models = set()
    return _available_models


def select_model(category: str) -> str | None:
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


def _to_b64(frame_bytes: bytes) -> str:
    return base64.b64encode(frame_bytes).decode("utf-8")


def query_model(
    frames_bytes: list[bytes],
    step:         dict,
    dish_name:    str,
    model:        str,
    category:     str,
) -> dict:
    """
    Send one or more frames to the vision model.
    Multi-frame inputs (motion steps) are all passed in the same message.
    """
    n_frames = len(frames_bytes)
    prompt   = build_prompt(step, dish_name, category, n_frames=n_frames)
    images   = [_to_b64(fb) for fb in frames_bytes]

    print(f"  [router] {model} | category={category} | frames={n_frames}")

    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role":    "user",
                "content": prompt,
                "images":  images,
            }],
            options={"temperature": 0.1},
        )
        raw = response["message"]["content"]
    except ollama.ResponseError as e:
        print(f"  [router] Model error: {e}")
        return {
            "done":     False,
            "feedback": "Couldn't get a clear read — keep going and I'll check again.",
            "model":    model,
            "category": category,
            "error":    True,
        }

    result             = parse_response(raw)
    result["model"]    = model
    result["category"] = category

    print(f"  [router] done={result['done']} | {result['feedback'][:60]}")
    return result


def check_frames(
    frames_bytes: list[bytes],
    step:         dict,
    dish_name:    str,
) -> dict:
    """Main entry point. Auto-selects model, queries, returns result."""
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
    print("\n  Vision routing:")
    for cat in sorted(MOONDREAM_CATEGORIES | LLAVA_CATEGORIES):
        model = select_model(cat)
        print(f"    {cat:14s} → {model or 'NOT AVAILABLE'}")
    print()
