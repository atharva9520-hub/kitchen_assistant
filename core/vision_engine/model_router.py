"""
core/vision_engine/model_router.py
-----------------------------------
Shared model router base.

Provides the shared query infrastructure:
  _get_available_models()  — queries Ollama once, caches result
  _to_b64()                — encode bytes to base64 string
  query_model()            — send frames + prompt to a model, parse result
  print_routing_table()    — print which model handles each category

select_model() and check_frames() are defined here as stubs.
Each profile's model_router.py overrides them with domain-specific
category→model routing tables.

Profiles:
  profiles/cooking/model_router.py   — colour/size/doneness → moondream/llava
  profiles/cocktail/model_router.py  — shake/stir → llava, pour/bottle → moondream
"""

import base64
import ollama
from prompts import build_prompt, parse_response, detect_category


MODEL_MOONDREAM = "moondream"
MODEL_LLAVA     = "llava"

_available_models: set[str] | None = None


# ── shared: model availability ────────────────────────────────────────────────

def _get_available_models() -> set[str]:
    """Query Ollama for installed models. Result cached after first call."""
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


# ── shared: encoding ──────────────────────────────────────────────────────────

def _to_b64(frame_bytes: bytes) -> str:
    return base64.b64encode(frame_bytes).decode("utf-8")


# ── shared: model query ───────────────────────────────────────────────────────

def query_model(
    frames_bytes: list[bytes],
    step:         dict,
    dish_name:    str,
    model:        str,
    category:     str,
) -> dict:
    """
    Send one or more frames to the vision model and return parsed result.
    Works for both single-frame (cooking) and multi-frame (cocktail shake/stir).

    Args:
        frames_bytes: list of JPEG byte strings (1 for single, 4+ for motion)
        step:         recipe step dict
        dish_name:    name of the dish or cocktail
        model:        ollama model name
        category:     check category (from detect_category)

    Returns:
        {
            "done":     bool,
            "feedback": str,
            "model":    str,
            "category": str,
        }
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


# ── stubs — overridden by profile ─────────────────────────────────────────────

def select_model(category: str) -> str | None:
    """
    Pick the best available model for a given category.
    Overridden by each profile's model_router.py.
    """
    raise NotImplementedError(
        "select_model() must be implemented in the active profile's model_router.py."
    )


def check_frames(
    frames_bytes: list[bytes],
    step:         dict,
    dish_name:    str,
) -> dict:
    """
    Main entry point: auto-select model, query, return result.
    Overridden by each profile's model_router.py.
    """
    raise NotImplementedError(
        "check_frames() must be implemented in the active profile's model_router.py."
    )


def print_routing_table():
    """Print category → model routing. Overridden by profile."""
    raise NotImplementedError(
        "print_routing_table() must be implemented in the active profile's model_router.py."
    )
