"""
prompts.py
----------
Builds vision model prompts tailored to each step type and check category.

The prompt is the most important part of the vision module.
A vague prompt → vague answer. A specific prompt → actionable teaching feedback.

Check categories (auto-detected from step content):
  "doneness"   — is it cooked through? (colour, texture, internal state)
  "colour"     — specific colour target (golden, translucent, brown)
  "size"       — chop/dice/slice size and evenness
  "texture"    — dough smoothness, sauce consistency, crisp vs soft
  "liquid"     — boiling, simmering, reduced, thickened
  "general"    — catch-all for anything else
"""

import re


# ── category detection ────────────────────────────────────────────────────────

CATEGORY_SIGNALS = {
    "colour": [
        "golden", "brown", "translucent", "caramelized", "pink", "white",
        "golden brown", "pale", "dark", "colour", "color",
    ],
    "size": [
        "chop", "dice", "slice", "mince", "julienne", "chunk", "piece",
        "size", "fine", "coarse", "thin", "thick", "even",
    ],
    "texture": [
        "smooth", "dough", "knead", "consistency", "thick", "creamy",
        "crispy", "crunchy", "tender", "soft", "firm",
    ],
    "liquid": [
        "boil", "simmer", "bubble", "reduce", "thicken", "sauce",
        "steam", "rolling boil", "gentle simmer",
    ],
    "doneness": [
        "cooked through", "done", "ready", "set", "cooked", "baked",
        "roasted", "grilled", "seared", "internal temperature",
    ],
}


def detect_category(step: dict) -> str:
    """
    Detect what kind of visual check this step needs.
    Returns one of: colour, size, texture, liquid, doneness, general.
    """
    text = " ".join([
        step.get("action", ""),
        step.get("visual_cue", ""),
        step.get("teaching", {}).get("watch_for", ""),
    ]).lower()

    # score each category
    scores = {cat: 0 for cat in CATEGORY_SIGNALS}
    for cat, signals in CATEGORY_SIGNALS.items():
        for signal in signals:
            if signal in text:
                scores[cat] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general"
    return best


# ── prompt templates ──────────────────────────────────────────────────────────

_BASE = """You are a patient cooking teacher watching a student cook {dish_name}.

Current step: {action}
Target: {visual_cue}

{category_instruction}

Respond in EXACTLY this format (no other text):
DONE: yes
FEEDBACK: <one teaching sentence>

or

DONE: no
FEEDBACK: <one teaching sentence describing what you see and what to watch for next>

Keep feedback under 25 words. Be specific about what you actually see."""


_CATEGORY_INSTRUCTIONS = {
    "colour": """Focus on the colour of the food.
Compare what you see against the target colour: {visual_cue}
Note: lighting can affect colour — look at the overall tone, not just highlights.""",

    "size": """Focus on the size and evenness of the cut pieces.
The target is: {visual_cue}
Look at: are pieces roughly equal in size? Are they the right thickness/width?
A rough guide — fine dice ~5mm, medium dice ~1cm, large dice ~2cm.""",

    "texture": """Focus on the surface texture and consistency.
The target is: {visual_cue}
Look at: surface smoothness, any visible lumps, how it holds its shape.""",

    "liquid": """Focus on the liquid's behaviour and state.
The target is: {visual_cue}
Look at: bubble size and frequency, steam, surface movement.
Gentle simmer = small bubbles at edges. Rolling boil = vigorous, large bubbles throughout.""",

    "doneness": """Focus on whether the food appears fully cooked.
The target is: {visual_cue}
Look at: colour, texture, any visible raw spots, how it holds together.""",

    "general": """Look carefully at the food and compare it to the target.
The target is: {visual_cue}
Describe what you see and whether it matches.""",
}


def build_prompt(step: dict, dish_name: str, category: str | None = None) -> str:
    """
    Build a targeted vision prompt for a given recipe step.

    Args:
        step:      step dict from recipe JSON
        dish_name: name of the dish being cooked
        category:  override auto-detection (colour/size/texture/liquid/doneness/general)

    Returns:
        prompt string ready to send to vision model
    """
    if category is None:
        category = detect_category(step)

    action = step.get("action", "cook the food")
    visual_cue = (
        step.get("visual_cue")
        or step.get("teaching", {}).get("watch_for", "")
        or "the step is complete"
    )

    category_instruction = _CATEGORY_INSTRUCTIONS.get(
        category, _CATEGORY_INSTRUCTIONS["general"]
    ).format(visual_cue=visual_cue)

    prompt = _BASE.format(
        dish_name=dish_name,
        action=action,
        visual_cue=visual_cue,
        category_instruction=category_instruction,
    )

    return prompt


def parse_response(text: str) -> dict:
    """
    Parse the vision model's response into a structured dict.

    Returns:
        {
            "done":     bool,
            "feedback": str,
            "raw":      str,   # original response
        }
    """
    text = text.strip()
    done = False
    feedback = text  # fallback if parsing fails

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("DONE:"):
            done = "yes" in line.lower()
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    return {
        "done":     done,
        "feedback": feedback,
        "raw":      text,
    }
