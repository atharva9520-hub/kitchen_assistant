"""
camera.py  (cocktail edition)
------------------------------
Handles both single-frame capture (body cam) and
multi-frame capture (fixed cam for motion steps).

Two cameras supported:
  CAMERA_BODY  (index 0) — body-mounted, ingredients + pours
  CAMERA_FIXED (index 1) — overhead fixed, shake + stir + muddle

Single-camera mode: set CAMERA_FIXED_ENABLED = False
  All checks fall back to body cam.
"""

import cv2
import time
import uuid
import json
import numpy as np
from pathlib import Path
from datetime import datetime


# ── config ────────────────────────────────────────────────────────────────────

CAMERA_BODY_INDEX    = 0
CAMERA_FIXED_INDEX   = 1
CAMERA_FIXED_ENABLED = False   # ← set True when second camera is connected

BURST_SECONDS  = 2.5
FRAMES_TO_GRAB = 10

# For motion steps: number of frames sampled across the clip
MOTION_FRAMES  = 4
MOTION_SECONDS = 10   # sample window for shake/stir

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

COLLECT_DATA = True
DATA_DIR     = Path(__file__).parent / "vision_data" / "frames"


# ── sharpness ─────────────────────────────────────────────────────────────────

def _sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _is_too_dark(frame: np.ndarray, threshold: float = 30.0) -> bool:
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) < threshold


# ── single-frame capture ──────────────────────────────────────────────────────

def capture_best_frame(camera_index: int = 0) -> tuple[np.ndarray | None, dict]:
    """
    Burst capture → pick sharpest frame.
    Used for: ingredient, pour, strain, garnish, presentation.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"  [camera] Could not open camera {camera_index}")
        return None, {"error": "camera_unavailable"}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # warm up
    for _ in range(3):
        cap.read()
        time.sleep(0.05)

    frames, scores = [], []
    interval = BURST_SECONDS / FRAMES_TO_GRAB

    for _ in range(FRAMES_TO_GRAB):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
            scores.append(_sharpness(frame))
        time.sleep(interval)

    cap.release()

    if not frames:
        return None, {"error": "no_frames"}

    best_frame, best_score = None, -1
    for frame, score in zip(frames, scores):
        if score > best_score and not _is_too_dark(frame):
            best_score, best_frame = score, frame

    if best_frame is None:
        best_score = max(scores)
        best_frame = frames[scores.index(best_score)]

    meta = {
        "timestamp":    datetime.now().isoformat(),
        "sharpness":    round(best_score, 1),
        "frames_taken": len(frames),
        "blur_warning": best_score < 50,
        "camera_index": camera_index,
        "mode":         "single",
    }

    if meta["blur_warning"]:
        print(f"  [camera] ⚠ Blurry frame (score={best_score:.0f})")
    print(f"  [camera] ✓ Single frame: sharpness={best_score:.0f}")
    return best_frame, meta


# ── multi-frame capture (motion steps) ───────────────────────────────────────

def capture_motion_frames(
    camera_index: int = 1,
    n_frames: int = MOTION_FRAMES,
    window_seconds: float = MOTION_SECONDS,
) -> tuple[list[np.ndarray], dict]:
    """
    Sample N evenly-spaced frames across a motion window.
    Used for: shake, stir, muddle.

    Returns (list_of_frames, metadata).
    Frames are returned in time order — the model sees the sequence.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        # fall back to body cam if fixed cam unavailable
        print(f"  [camera] Camera {camera_index} unavailable — falling back to body cam")
        cap = cv2.VideoCapture(CAMERA_BODY_INDEX)
        camera_index = CAMERA_BODY_INDEX
        if not cap.isOpened():
            return [], {"error": "camera_unavailable"}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # warm up
    for _ in range(3):
        cap.read()
        time.sleep(0.05)

    interval = window_seconds / n_frames
    sampled  = []

    print(f"  [camera] Motion capture: {n_frames} frames over {window_seconds}s…")

    for i in range(n_frames):
        time.sleep(interval)
        ret, frame = cap.read()
        if ret and frame is not None:
            sampled.append(frame)
            print(f"  [camera] Frame {i+1}/{n_frames} captured")

    cap.release()

    meta = {
        "timestamp":    datetime.now().isoformat(),
        "frames_taken": len(sampled),
        "window_seconds": window_seconds,
        "camera_index": camera_index,
        "mode":         "motion",
    }

    print(f"  [camera] ✓ Motion capture: {len(sampled)} frames")
    return sampled, meta


# ── camera routing ────────────────────────────────────────────────────────────

MOTION_CATEGORIES = {"shake", "stir", "muddle"}

def capture_for_step(step: dict, category: str) -> tuple[list[np.ndarray], dict]:
    """
    Route to the right capture method based on step category and camera config.

    Returns (frames, metadata) — always a list for consistency.
    Single-frame checks return a 1-element list.
    """
    cam_assignment = step.get("camera", "body")

    # motion steps → multi-frame from fixed cam (or body if fixed unavailable)
    if category in MOTION_CATEGORIES:
        if CAMERA_FIXED_ENABLED:
            frames, meta = capture_motion_frames(CAMERA_FIXED_INDEX)
        else:
            print("  [camera] Fixed cam disabled — using body cam for motion step")
            frames, meta = capture_motion_frames(CAMERA_BODY_INDEX)
        return frames, meta

    # presentation → body cam (could be both in future)
    # everything else → body cam
    frame, meta = capture_best_frame(CAMERA_BODY_INDEX)
    return ([frame] if frame is not None else []), meta


# ── encoding ──────────────────────────────────────────────────────────────────

def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

def frames_to_jpeg_bytes(frames: list[np.ndarray], quality: int = 85) -> list[bytes]:
    return [frame_to_jpeg_bytes(f, quality) for f in frames]


# ── data collection ───────────────────────────────────────────────────────────

def save_training_frame(
    frames: list[np.ndarray],
    step: dict,
    result: dict,
    dish_name: str,
    frame_meta: dict,
):
    if not COLLECT_DATA or not frames:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dish_slug = dish_name.lower().replace(" ", "_")[:20]
    step_num  = step.get("step_number", 0)
    uid       = uuid.uuid4().hex[:8]
    base_name = f"{dish_slug}_step{step_num}_{uid}"

    # save all frames
    for i, frame in enumerate(frames):
        img_path = DATA_DIR / f"{base_name}_f{i}.jpg"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # save label
    label = {
        "dish_name":    dish_name,
        "step_number":  step_num,
        "action":       step.get("action", ""),
        "visual_cue":   step.get("visual_cue", ""),
        "trigger_type": step.get("trigger_type", ""),
        "camera":       step.get("camera", ""),
        "n_frames":     len(frames),
        "vision_result": result,
        "frame_meta":   frame_meta,
        "collected_at": datetime.now().isoformat(),
    }

    with open(DATA_DIR / f"{base_name}.json", "w") as f:
        json.dump(label, f, indent=2)

    print(f"  [camera] 💾 Saved {len(frames)} training frame(s) → {base_name}")
