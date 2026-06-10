"""
normalizer.py
-------------
Takes raw extracted text and turns it into a clean, structured recipe
using a local Ollama LLM (Mistral 7B or similar).

Output schema (list of steps):
[
  {
    "step_number": 1,
    "action": "Dice the onion",
    "ingredients": ["1 onion"],
    "duration_minutes": null,       # null means unknown / not applicable
    "temperature": null,            # e.g. "medium heat", "180C", null
    "visual_cue": null,             # e.g. "until golden", null
    "technique_note": null,         # e.g. "cut against the grain", null
    "gaps": ["duration", "temp"]    # fields the gap detector should fill
  },
  ...
]

Also returns top-level metadata:
{
  "dish_name": str,
  "cuisine": str | null,
  "servings": str | null,
  "ingredients": [str],   # master ingredient list
  "steps": [...]
}
"""

import json
import re
import ollama


# ── prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a recipe parsing assistant. Your job is to convert 
raw recipe text (which may be a video transcript, social media caption, or 
plain text) into a clean, structured JSON recipe.

Rules:
- Extract every cooking step in order.
- For each step, extract: action, ingredients used, duration, temperature, 
  visual cue (what it should look like when done), and technique notes.
- If a field is not mentioned, set it to null and add the field name to "gaps".
- Be generous with gaps — if a duration seems implied but not stated, mark it.
- Consolidate repeated mentions. Do not duplicate steps.
- Output ONLY valid JSON, no markdown, no explanation.
"""

NORMALIZE_PROMPT = """Convert this recipe text into structured JSON.

Recipe text:
\"\"\"
{raw_text}
\"\"\"

Return this exact JSON structure:
{{
  "dish_name": "name of the dish",
  "cuisine": "cuisine type or null",
  "servings": "e.g. 4 or null",
  "ingredients": ["full ingredient list as strings"],
  "steps": [
    {{
      "step_number": 1,
      "action": "what to do",
      "ingredients": ["ingredients used in this step"],
      "duration_minutes": null,
      "temperature": null,
      "visual_cue": null,
      "technique_note": null,
      "gaps": []
    }}
  ]
}}

Only output JSON. No markdown fences, no explanation.
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean_json_response(text: str) -> str:
    """Strip markdown fences if the LLM added them anyway."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


def _call_ollama(prompt: str, model: str = "mistral") -> str:
    """Call local Ollama and return response text."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        options={"temperature": 0.1},  # low temp for structured output
    )
    return response["message"]["content"]


def _infer_gaps(step: dict) -> list[str]:
    """
    Double-check gaps list — add any null fields the LLM missed flagging.
    Only flags fields that are actually useful to fill for cooking.
    """
    gaps = list(step.get("gaps", []))
    action = step.get("action", "").lower()

    # these actions almost always have a meaningful duration
    time_sensitive = ["cook", "simmer", "boil", "fry", "sauté", "bake",
                      "roast", "grill", "steam", "marinate", "rest", "chill",
                      "reduce", "caramelize", "brown"]

    needs_time = any(word in action for word in time_sensitive)

    if needs_time and step.get("duration_minutes") is None and "duration" not in gaps:
        gaps.append("duration")

    if needs_time and step.get("temperature") is None and "temperature" not in gaps:
        gaps.append("temperature")

    return gaps


# ── public API ────────────────────────────────────────────────────────────────

def normalize(extracted: dict, model: str = "mistral") -> dict:
    """
    Takes extractor output dict, returns normalized recipe dict.
    
    extracted: output from extractor.extract()
    model: Ollama model name to use
    """
    raw_text = extracted["raw_text"]
    title_hint = extracted.get("title") or ""
    description = extracted.get("description") or ""

    # combine sources for richer context
    full_input = raw_text
    if description and description not in raw_text:
        full_input = f"Video title: {title_hint}\n\nDescription:\n{description}\n\nTranscript:\n{raw_text}"

    # truncate if too long (Mistral context window safe limit)
    if len(full_input) > 6000:
        full_input = full_input[:6000] + "\n[truncated]"

    print(f"[normalizer] Running LLM normalization with {model}…")
    prompt = NORMALIZE_PROMPT.format(raw_text=full_input)
    raw_response = _call_ollama(prompt, model)

    try:
        cleaned = _clean_json_response(raw_response)
        recipe = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON: {e}\nResponse was:\n{raw_response[:500]}"
        )

    # post-process: tighten up gaps list on each step
    for step in recipe.get("steps", []):
        step["gaps"] = _infer_gaps(step)

    print(f"[normalizer] ✓ {len(recipe.get('steps', []))} steps extracted")
    return recipe
