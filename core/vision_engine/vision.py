"""
vision.py
---------
Handles camera capture and vision model evaluation.
Called by the timer engine when a condition step needs checking.

Currently a STUB — returns a simulated response so the timer engine
works end-to-end without a camera connected.

To activate real vision:
  1. Install ollama and pull moondream:  ollama pull moondream
  2. Install opencv:                     pip install opencv-python
  3. Set VISION_ENABLED = True below

The stub makes it easy to test the full cooking session flow
before the camera hardware is connected.
"""

import base64
import subprocess
import tempfile
import os
import ollama

# ── config ────────────────────────────────────────────────────────────────────

VISION_ENABLED  = False       # set True when camera + moondream are ready
VISION_MODEL    = "moondream" # or "llava" for better accuracy
CAMERA_INDEX    = 0           # 0 = built-in webcam, change for external
CAPTURE_SECONDS = 2.5         # how long to capture before picking best frame
FRAMES_TO_GRAB  = 8           # frames captured in the burst


# ── camera capture ────────────────────────────────────────────────────────────

def _capture_best_frame() -> str | None:
    """
    Captures a short burst from the camera and picks the sharpest frame.
    Returns path to saved JPEG, or None on failure.

    Sharpness = Laplacian variance (standard blur detection).
    Higher = sharper.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  [vision] opencv not installed — run: pip install opencv-python")
        return None

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"  [vision] Could not open camera {CAMERA_INDEX}")
        return None

    frames = []
    for _ in range(FRAMES_TO_GRAB):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    if not frames:
        print("  [vision] No frames captured")
        return None

    # pick sharpest frame using Laplacian variance
    def sharpness(f):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    best = max(frames, key=sharpness)

    # save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, best)
    return tmp.name


def _frame_to_base64(image_path: str) -> str:
    """Read image file and return base64 string for ollama."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── vision model ──────────────────────────────────────────────────────────────

VISION_PROMPT = """You are a cooking assistant checking on a student's progress.

Recipe context: {dish_name}
Current step: {action}
What we're looking for: {visual_cue}

Look at the image carefully. Answer two things:
1. Is the condition met? (yes/no)
2. One sentence of teaching feedback describing what you see and what to do next.

Format your response exactly as:
DONE: yes
FEEDBACK: <one sentence>

or

DONE: no  
FEEDBACK: <one sentence describing current state and what to watch for>
"""


def _ask_vision_model(image_path: str, step: dict, dish_name: str) -> dict:
    """
    Send a frame to the vision model with a cooking-specific prompt.
    Returns {"done": bool, "feedback": str}
    """
    visual_cue = (
        step.get("visual_cue")
        or step.get("teaching", {}).get("watch_for", "")
        or "the step is complete"
    )

    prompt = VISION_PROMPT.format(
        dish_name=dish_name,
        action=step.get("action", ""),
        visual_cue=visual_cue,
    )

    image_b64 = _frame_to_base64(image_path)

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_b64],
        }],
        options={"temperature": 0.1},
    )

    text = response["message"]["content"].strip()

    # parse response
    done     = False
    feedback = text  # fallback

    for line in text.splitlines():
        if line.upper().startswith("DONE:"):
            done = "yes" in line.lower()
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    return {"done": done, "feedback": feedback}


# ── stub (used when VISION_ENABLED = False) ───────────────────────────────────

_stub_poll_count: dict[int, int] = {}  # step_number → poll count

def _stub_check(step: dict) -> dict:
    """
    Simulates vision responses for testing without a camera.
    Returns 'not done' for the first 2 polls, then 'done'.
    """
    n = step.get("step_number", 0)
    _stub_poll_count[n] = _stub_poll_count.get(n, 0) + 1
    count = _stub_poll_count[n]

    action     = step.get("action", "this")
    visual_cue = step.get("visual_cue") or "done"

    if count <= 2:
        return {
            "done": False,
            "feedback": (
                f"[STUB] Still working on {action}. "
                f"Not yet {visual_cue}. Keep going."
            ),
        }
    else:
        return {
            "done": True,
            "feedback": (
                f"[STUB] {action} looks great — {visual_cue}. "
                f"Ready for the next step."
            ),
        }


# ── public API ────────────────────────────────────────────────────────────────

def check_condition(step: dict, dish_name: str) -> dict:
    """
    Main entry point. Captures a frame and evaluates whether
    the step's visual condition is met.

    Returns:
        {
            "done":     bool,   # True if condition is met
            "feedback": str,    # teaching sentence to speak aloud
        }
    """
    if not VISION_ENABLED:
        print("  [vision] Stub mode — no camera")
        return _stub_check(step)

    print("  [vision] Capturing frame…")
    image_path = _capture_best_frame()

    if not image_path:
        return {
            "done": False,
            "feedback": "I couldn't get a clear view — make sure the camera is pointed at the food.",
        }

    try:
        result = _ask_vision_model(image_path, step, dish_name)
    finally:
        # clean up temp file
        try:
            os.unlink(image_path)
        except Exception:
            pass

    print(f"  [vision] done={result['done']} | {result['feedback'][:60]}")
    return result
