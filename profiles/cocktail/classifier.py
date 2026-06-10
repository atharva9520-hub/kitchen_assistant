"""
classifier.py  (cocktail edition)
----------------------------------
Labels each cocktail recipe step with a trigger type and camera assignment.

Trigger types:
  "timed"       — fixed duration (stir 30 sec, shake 12 sec)
  "condition"   — needs visual confirmation (until branch melts, until cloudy)
  "instruction" — one-shot action (add ingredient, garnish, serve)

Camera assignments (set alongside trigger type):
  "body"        — body-mounted cam (ingredients, pours, garnish)
  "fixed"       — overhead fixed cam (shake, stir, muddle — motion steps)
  "both"        — final presentation check
  "none"        — instruction-only, no camera needed

Step categories (cocktail-specific):
  ingredient, pour, shake, stir, muddle, strain, garnish, presentation, instruction
"""

import json
import re
import ollama


# ── rule-based pre-classifier ─────────────────────────────────────────────────

TIMED_VERBS = {
    "shake", "stir", "muddle", "churn", "spin", "swizzle",
    "chill", "rest", "infuse", "carbonate",
}

CONDITION_PHRASES = {
    "until", "when", "once", "till", "as soon as",
    "diluted", "chilled", "cloudy", "clear", "frosted",
    "branch melts", "ice melts", "properly mixed",
}

INSTRUCTION_VERBS = {
    "add", "pour", "measure", "jigger", "fill", "top",
    "garnish", "express", "twist", "rim", "strain", "double strain",
    "fine strain", "serve", "place", "drop", "float", "layer",
    "sprinkle", "grate", "pick", "skewer", "insert",
}

# Camera assignment by first verb in action
FIXED_CAM_VERBS  = {"shake", "stir", "muddle", "churn", "spin", "swizzle"}
BODY_CAM_VERBS   = {"pour", "measure", "add", "garnish", "express", "twist",
                    "rim", "strain", "double strain", "fine strain", "float", "layer"}


def _rule_based_classify(step: dict) -> tuple[str | None, str | None]:
    """
    Returns (trigger_type, camera) or (None, None) if ambiguous.
    """
    action    = step.get("action", "").lower()
    cue       = step.get("visual_cue", "") or ""
    duration  = step.get("duration_minutes")
    first     = action.split()[0] if action else ""

    # camera assignment from first verb
    if first in FIXED_CAM_VERBS:
        camera = "fixed"
    elif first in BODY_CAM_VERBS:
        camera = "body"
    else:
        camera = None  # let LLM decide

    # trigger type
    if duration and first in TIMED_VERBS:
        return "timed", camera or "fixed"

    if cue or any(p in action + cue for p in CONDITION_PHRASES):
        return "condition", camera or "body"

    if first in INSTRUCTION_VERBS and not duration and not cue:
        return "instruction", camera or "none"

    return None, camera


# ── LLM classification ────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """You are a cocktail teacher classifying recipe steps for a 
real-time teaching assistant.

For this step decide:

1. trigger_type:
   - "timed"       — has a fixed duration the student must time
   - "condition"   — student watches for a visual/sensory cue to know when done
   - "instruction" — one-shot action, no waiting

2. camera:
   - "body"   — body-mounted cam (ingredient ID, pour, garnish, strain)
   - "fixed"  — overhead fixed cam (shake, stir, muddle — motion evaluation)
   - "both"   — final glass presentation
   - "none"   — no camera needed (e.g. just 'serve')

3. poll_interval_seconds: for "condition" steps only (null otherwise)
   Typical: 10s for fast things (ice branch melt), 20s for medium (dilution check)

4. teaching content — explain like a patient bar teacher:
   - why: why this step matters to the cocktail
   - watch_for: what to look, feel, or hear for
   - common_mistake: what beginners get wrong

Cocktail: {dish_name}
Step {step_number}: {action}
Duration: {duration}
Visual cue: {visual_cue}
Technique note: {technique_note}

Respond with ONLY this JSON (no markdown):
{{
  "trigger_type": "timed" | "condition" | "instruction",
  "camera": "body" | "fixed" | "both" | "none",
  "poll_interval_seconds": <number or null>,
  "teaching": {{
    "why": "...",
    "watch_for": "...",
    "common_mistake": "..."
  }}
}}
"""


def _llm_classify(step: dict, dish_name: str, model: str) -> dict:
    prompt = CLASSIFY_PROMPT.format(
        dish_name=dish_name,
        step_number=step.get("step_number", "?"),
        action=step.get("action", ""),
        duration=step.get("duration_string") or step.get("duration_minutes") or "not specified",
        visual_cue=step.get("visual_cue") or "none",
        technique_note=step.get("technique_note") or "none",
    )

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    raw = response["message"]["content"].strip()
    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        trigger = "instruction"
        if "timed"     in raw: trigger = "timed"
        elif "condition" in raw: trigger = "condition"
        cam = "body"
        if "fixed" in raw: cam = "fixed"
        elif "both"  in raw: cam = "both"
        elif "none"  in raw: cam = "none"
        return {
            "trigger_type": trigger,
            "camera": cam,
            "poll_interval_seconds": None,
            "teaching": {"why": "", "watch_for": "", "common_mistake": ""},
        }


# ── public API ────────────────────────────────────────────────────────────────

def classify(recipe: dict, model: str = "mistral") -> dict:
    dish_name = recipe.get("dish_name", "this cocktail")
    steps     = recipe.get("steps", [])

    print(f"[classifier] Classifying {len(steps)} steps…")

    for step in steps:
        action = step.get("action", "")
        trigger_type, camera = _rule_based_classify(step)

        if trigger_type and trigger_type != "condition":
            llm_result = _llm_classify(step, dish_name, model)
            step["trigger_type"]          = trigger_type
            step["camera"]                = camera or llm_result.get("camera", "body")
            step["poll_interval_seconds"] = None
            step["teaching"]              = llm_result.get("teaching", {})
        else:
            llm_result = _llm_classify(step, dish_name, model)
            step["trigger_type"]          = llm_result.get("trigger_type", "instruction")
            step["camera"]                = camera or llm_result.get("camera", "body")
            step["poll_interval_seconds"] = llm_result.get("poll_interval_seconds")
            step["teaching"]              = llm_result.get("teaching", {})

        if step["trigger_type"] == "condition" and not step.get("poll_interval_seconds"):
            step["poll_interval_seconds"] = 15  # cocktail checks are fast

        cam_label = step.get("camera", "?")
        print(
            f"  Step {step['step_number']}: [{step['trigger_type']:11s}]"
            f" [cam:{cam_label:5s}] {action[:45]}"
        )

    recipe["steps"] = steps
    return recipe
