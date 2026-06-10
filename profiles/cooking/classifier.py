"""
classifier.py
-------------
Labels each recipe step with a trigger type for the cooking assistant:

  "timed"         — has a fixed duration; fire an alert when time is up
  "condition"     — needs visual confirmation; poll camera periodically
  "instruction"   — one-shot action; read aloud and move on

Also enriches each step with teaching content — a short explanation of
WHY the step matters, what to watch for, and common mistakes.

Output adds to each step:
  {
    "trigger_type": "timed" | "condition" | "instruction",
    "poll_interval_seconds": 30 | null,   # for condition steps only
    "teaching": {
        "why": "explanation of why this step matters",
        "watch_for": "what to look/smell/feel for",
        "common_mistake": "what beginners often get wrong"
    }
  }
"""

import json
import re
import ollama


# ── rule-based pre-classifier (fast, no LLM) ─────────────────────────────────

# If any of these appear and we have a duration → timed
TIMED_VERBS = {
    "simmer", "boil", "bake", "roast", "grill", "fry", "sauté", "sear",
    "steam", "braise", "poach", "marinate", "rest", "chill", "freeze",
    "refrigerate", "reduce", "caramelize", "brown", "toast", "blanch",
    "deep fry", "stir fry", "pan fry", "slow cook", "pressure cook",
}

# If any of these appear → likely condition (visual check needed)
CONDITION_PHRASES = {
    "until", "when", "once", "till", "as soon as",
    "golden", "translucent", "tender", "caramelized", "brown",
    "soft", "crispy", "bubbling", "reduced", "thickened",
    "fragrant", "aromatic", "set", "cooked through", "done",
}

# Always instruction-only regardless
INSTRUCTION_VERBS = {
    "add", "season", "sprinkle", "pour", "transfer", "place",
    "remove", "drain", "strain", "slice", "dice", "chop", "mince",
    "grate", "peel", "wash", "rinse", "crush", "mix", "stir",
    "combine", "whisk", "fold", "plate", "garnish", "serve",
}


def _rule_based_classify(step: dict) -> str | None:
    """
    Fast rule-based pre-classification. Returns trigger type or None
    if ambiguous (LLM will handle those).
    """
    action = step.get("action", "").lower()
    visual_cue = step.get("visual_cue") or ""
    duration = step.get("duration_minutes")

    words = set(re.findall(r"\w+", action + " " + visual_cue))

    # has a duration + timed verb → timed
    if duration and any(v in action for v in TIMED_VERBS):
        return "timed"

    # has a visual cue or condition phrase → condition
    if visual_cue or any(p in action + visual_cue for p in CONDITION_PHRASES):
        return "condition"

    # pure instruction verbs with no duration/cue → instruction
    first_word = action.split()[0] if action else ""
    if first_word in INSTRUCTION_VERBS and not duration and not visual_cue:
        return "instruction"

    return None  # ambiguous — send to LLM


# ── LLM classification + teaching enrichment ─────────────────────────────────

CLASSIFY_PROMPT = """You are a cooking teacher classifying recipe steps.

For each step, decide:
1. trigger_type: 
   - "timed" if the step has a specific duration and the timer should fire an alert
   - "condition" if the user needs to watch for a visual/sensory cue to know when done
   - "instruction" if it's a one-shot action with no waiting
   
2. poll_interval_seconds: for "condition" steps only — how often (in seconds) 
   should the camera check? Typical: 20 for fast things (oil shimmer), 
   30-45 for medium (onions softening), 60 for slow (reducing sauce).
   Set to null for timed/instruction.

3. teaching content — explain like a patient cooking teacher:
   - why: why this step matters to the dish
   - watch_for: what to look, smell, or feel for
   - common_mistake: what beginners most often get wrong here

Recipe: {dish_name}
Step {step_number}: {action}
Duration: {duration}
Temperature: {temperature}
Visual cue: {visual_cue}
Technique note: {technique_note}

Respond with ONLY this JSON (no markdown, no explanation):
{{
  "trigger_type": "timed" | "condition" | "instruction",
  "poll_interval_seconds": <number or null>,
  "teaching": {{
    "why": "...",
    "watch_for": "...",
    "common_mistake": "..."
  }}
}}
"""


def _llm_classify(step: dict, dish_name: str, model: str) -> dict:
    """Use LLM to classify ambiguous steps and generate teaching content."""
    prompt = CLASSIFY_PROMPT.format(
        dish_name=dish_name,
        step_number=step.get("step_number", "?"),
        action=step.get("action", ""),
        duration=step.get("duration_string") or step.get("duration_minutes") or "not specified",
        temperature=step.get("temperature") or "not specified",
        visual_cue=step.get("visual_cue") or "none",
        technique_note=step.get("technique_note") or "none",
    )

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    raw = response["message"]["content"].strip()

    # strip markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # fallback: extract what we can
        trigger = "instruction"
        if "timed" in raw:
            trigger = "timed"
        elif "condition" in raw:
            trigger = "condition"
        return {
            "trigger_type": trigger,
            "poll_interval_seconds": None,
            "teaching": {
                "why": "",
                "watch_for": "",
                "common_mistake": "",
            },
        }


# ── poll interval heuristic ───────────────────────────────────────────────────

def _default_poll_interval(step: dict) -> int:
    """Assign a sensible poll interval for condition steps if LLM didn't."""
    action = step.get("action", "").lower()

    fast_conditions = ["oil", "butter", "shimmer", "smoke", "foam"]
    slow_conditions = ["reduce", "braise", "simmer", "slow"]

    if any(w in action for w in fast_conditions):
        return 15
    if any(w in action for w in slow_conditions):
        return 60
    return 30  # default


# ── public API ────────────────────────────────────────────────────────────────

def classify(recipe: dict, model: str = "mistral") -> dict:
    """
    Takes a gap-filled recipe dict, adds trigger_type and teaching 
    content to every step. Returns updated recipe dict.
    """
    dish_name = recipe.get("dish_name", "this dish")
    steps = recipe.get("steps", [])

    print(f"[classifier] Classifying {len(steps)} steps…")

    for step in steps:
        action = step.get("action", "")

        # try fast rule-based first
        trigger_type = _rule_based_classify(step)

        if trigger_type and trigger_type != "condition":
            # rule-based got a clear answer — still run LLM for teaching content
            llm_result = _llm_classify(step, dish_name, model)
            step["trigger_type"] = trigger_type  # trust rule over LLM here
            step["poll_interval_seconds"] = None
            step["teaching"] = llm_result.get("teaching", {})
        else:
            # ambiguous or condition — let LLM decide everything
            llm_result = _llm_classify(step, dish_name, model)
            step["trigger_type"] = llm_result.get("trigger_type", "instruction")
            step["poll_interval_seconds"] = llm_result.get("poll_interval_seconds")
            step["teaching"] = llm_result.get("teaching", {})

        # ensure condition steps always have a poll interval
        if step["trigger_type"] == "condition" and not step["poll_interval_seconds"]:
            step["poll_interval_seconds"] = _default_poll_interval(step)

        print(
            f"  Step {step['step_number']}: [{step['trigger_type']:11s}] {action[:50]}"
        )

    recipe["steps"] = steps
    return recipe
