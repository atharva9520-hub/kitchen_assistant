"""
prompts.py  (cocktail edition)
------------------------------
Builds vision model prompts tailored to cocktail step categories.

Categories:
  ingredient    — is this the right bottle / ingredient?
  pour          — correct volume, colour in glass
  shake         — vigour, duration, technique (fixed cam, multi-frame)
  stir          — rotation speed, ice contact, dilution (fixed cam, multi-frame)
  muddle        — pressure, coverage (fixed cam)
  strain        — technique, clarity of pour
  garnish       — correct garnish, placement, expression
  presentation  — final glass appearance, colour, clarity
  general       — catch-all
"""

import re


# ── category detection ────────────────────────────────────────────────────────

CATEGORY_SIGNALS = {
    "shake":       ["shake", "shaken", "shaker", "tin"],
    "stir":        ["stir", "stirred", "mixing glass", "bar spoon", "rotation"],
    "muddle":      ["muddle", "muddler", "crush", "press"],
    "pour":        ["pour", "measure", "jigger", "oz", "ml", "cl", "top", "float", "layer"],
    "ingredient":  ["add", "grab", "pick", "use", "select", "bottle", "ingredient"],
    "strain":      ["strain", "fine strain", "double strain", "hawthorne", "julep"],
    "garnish":     ["garnish", "twist", "express", "rim", "grate", "zest", "wedge",
                    "slice", "cherry", "olive", "pick", "skewer", "sprig", "wheel"],
    "presentation":["serve", "final", "glass", "appearance", "presentation"],
}


def detect_category(step: dict) -> str:
    text = " ".join([
        step.get("action", ""),
        step.get("visual_cue", "") or "",
        step.get("teaching", {}).get("watch_for", "") or "",
    ]).lower()

    scores = {cat: 0 for cat in CATEGORY_SIGNALS}
    for cat, signals in CATEGORY_SIGNALS.items():
        for sig in signals:
            if sig in text:
                scores[cat] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ── prompt templates ──────────────────────────────────────────────────────────

_BASE = """You are an experienced bartender teaching a student to make {dish_name}.

Current step: {action}
Target: {visual_cue}

{category_instruction}

Respond in EXACTLY this format:
DONE: yes
FEEDBACK: <one teaching sentence under 25 words>

or

DONE: no
FEEDBACK: <one sentence: what you see + what to do next>"""


_CATEGORY_INSTRUCTIONS = {

    "ingredient": """Look at the bottle or ingredient the student is holding.
Target: {visual_cue}
Check: Is this the correct bottle? Read the label if visible.
Look for brand name, bottle shape, and liquid colour as identifiers.""",

    "pour": """Look at the liquid in the glass or jigger.
Target: {visual_cue}
Check: Does the volume and colour look correct?
A standard jigger is 1.5oz / 45ml. The liquid line and colour are your cues.""",

    "shake": """You are looking at {n_frames} frames from a shaking sequence.
Target: {visual_cue}
Check across the frames:
- Is the shaker moving with vigour? (big, energetic strokes — not timid)
- Is the motion consistent? (not stopping between strokes)
- A proper shake is 10-15 seconds of hard work.
Look for condensation forming on the tin as a sign of proper chilling.""",

    "stir": """You are looking at {n_frames} frames from a stirring sequence.
Target: {visual_cue}
Check across the frames:
- Is the bar spoon rotating smoothly around the inside of the glass?
- Is the ice making full contact with the liquid?
- Proper stirring is slow, circular, and controlled — not fast or splashy.
Look for the liquid surface rotating as one unified mass.""",

    "muddle": """Look at the muddled ingredients in the glass.
Target: {visual_cue}
Check: Are the ingredients properly broken down?
Citrus should be pressed to release juice but not shredded (releases bitter pith).
Herbs should be lightly bruised, not pulverised.""",

    "strain": """Look at the liquid being strained into the glass.
Target: {visual_cue}
Check: Is the pour clean and controlled?
Look for: no ice chips, consistent pour rate, strainer held snugly against the tin.""",

    "garnish": """Look at the garnish being applied to the glass.
Target: {visual_cue}
Check: Is the garnish correct and well-placed?
For citrus twists: look for the expressed oils catching the light (shiny mist).
For herbs: should look fresh and upright, not wilted or submerged.""",

    "presentation": """Look at the finished cocktail.
Target: {visual_cue}
Evaluate the final glass:
- Colour: does it match what the cocktail should look like?
- Clarity: should it be clear, cloudy, or layered?
- Garnish: properly placed?
- Glass: correct style, clean, chilled if needed?""",

    "general": """Look at what the student is doing.
Target: {visual_cue}
Compare what you see to the target and give a brief, honest assessment.""",
}


def build_prompt(
    step: dict,
    dish_name: str,
    category: str | None = None,
    n_frames: int = 1,
) -> str:
    if category is None:
        category = detect_category(step)

    action = step.get("action", "prepare the cocktail")
    visual_cue = (
        step.get("visual_cue")
        or step.get("teaching", {}).get("watch_for", "")
        or "the step is done correctly"
    )

    cat_instruction = _CATEGORY_INSTRUCTIONS.get(
        category, _CATEGORY_INSTRUCTIONS["general"]
    ).format(visual_cue=visual_cue, n_frames=n_frames)

    return _BASE.format(
        dish_name=dish_name,
        action=action,
        visual_cue=visual_cue,
        category_instruction=cat_instruction,
    )


def parse_response(text: str) -> dict:
    text = text.strip()
    done = False
    feedback = text

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("DONE:"):
            done = "yes" in line.lower()
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    return {"done": done, "feedback": feedback, "raw": text}
