"""
core/vision_engine/prompts.py
------------------------------
Shared prompt base.

Only parse_response() lives here — it's identical in both profiles.

detect_category() and build_prompt() are defined in each profile's
prompts.py and override this base via sys.path injection in run.py.

Profiles:
  profiles/cooking/prompts.py   — colour, size, texture, doneness, liquid
  profiles/cocktail/prompts.py  — shake, stir, pour, ingredient, garnish
"""


def parse_response(text: str) -> dict:
    """
    Parse a vision model response into a structured dict.

    Expects the model to respond in this format:
        DONE: yes
        FEEDBACK: one teaching sentence

    Returns:
        {
            "done":     bool,
            "feedback": str,
            "raw":      str,
        }
    """
    text     = text.strip()
    done     = False
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


# ── stubs — must be overridden by profile ────────────────────────────────────

def detect_category(step: dict) -> str:
    """
    Detect what kind of visual check this step needs.
    Overridden by each profile's prompts.py.
    """
    raise NotImplementedError(
        "detect_category() must be implemented in the active profile's prompts.py. "
        "Make sure run.py has added the profile path to sys.path."
    )


def build_prompt(
    step: dict,
    dish_name: str,
    category: str | None = None,
    **kwargs,
) -> str:
    """
    Build a vision model prompt for this step.
    Overridden by each profile's prompts.py.
    """
    raise NotImplementedError(
        "build_prompt() must be implemented in the active profile's prompts.py."
    )
