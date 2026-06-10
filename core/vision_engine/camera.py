"""
core/vision_engine/camera.py
-----------------------------
Shared camera base. Hardware-level only — no domain logic.

Provides:
  capture_best_frame()     — burst capture, returns sharpest single frame
  frame_to_jpeg_bytes()    — encode frame for model input
  frames_to_jpeg_bytes()   — encode multiple frames
  save_training_frame()    — persist frame + label for future training
  _sharpness()             — Laplacian variance score
  _is_too_dark()           — brightness guard

Profiles override this file when they need extra behaviour:
  cocktail profile adds capture_motion_frames() and capture_for_step()
  cooking profile uses this base as-is
"""

import cv2
import time
import uuid
import json
import numpy as np
from pathlib import Path
from datetime import datetime


# ── config ────────────────────────────────────────────────────────────────────

CAMERA_BODY_INDEX = 0      # built-in or first external webcam
BURST_SECONDS     = 2.5
FRAMES_TO_GRAB    = 10
FRAME_WIDTH       = 1280
FRAME_HEIGHT      = 720

COLLECT_DATA = True
DATA_DIR     = Path(__file__).parent.parent.parent / "vision_data" / "frames"


# ── frame quality ─────────────────────────────────────────────────────────────

def _sharpness(frame: np.ndarray) -> float:
    """Laplacian variance — higher = sharper. Below ~50 is too blurry."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _is_too_dark(frame: np.ndarray, threshold: float = 30.0) -> bool:
    """True if mean brightness is below threshold."""
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) < threshold


# ── single-frame burst capture ────────────────────────────────────────────────

def capture_best_frame(
    camera_index: int = CAMERA_BODY_INDEX,
) -> tuple[np.ndarray | None, dict]:
    """
    Capture a burst of frames, return the sharpest usable one.

    Returns:
        (frame, metadata)  — frame is a numpy BGR array or None on failure
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"  [camera] Could not open camera {camera_index}")
        return None, {"error": "camera_unavailable"}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # warm up — first frames from webcam are often underexposed
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

    # pick sharpest non-dark frame
    best_frame, best_score = None, -1
    for frame, score in zip(frames, scores):
        if score > best_score and not _is_too_dark(frame):
            best_score, best_frame = score, frame

    # fallback: all dark — take sharpest anyway
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
    else:
        print(f"  [camera] ✓ Frame captured (sharpness={best_score:.0f})")

    return best_frame, meta


# ── encoding ──────────────────────────────────────────────────────────────────

def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a single numpy frame to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def frames_to_jpeg_bytes(
    frames: list[np.ndarray], quality: int = 85
) -> list[bytes]:
    """Encode a list of frames to JPEG bytes."""
    return [frame_to_jpeg_bytes(f, quality) for f in frames]


def frame_to_jpeg_file(frame: np.ndarray, path: str, quality: int = 85):
    """Save a frame to disk as JPEG."""
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])


# ── training data collection ──────────────────────────────────────────────────

def save_training_frame(
    frames:     list[np.ndarray],
    step:       dict,
    result:     dict,
    dish_name:  str,
    frame_meta: dict,
):
    """
    Save captured frames + vision result label for future model training.

    Directory: vision_data/frames/
    Files per capture:
        <dish>_step<N>_<uid>_f<i>.jpg   — each frame
        <dish>_step<N>_<uid>.json       — metadata + label
    """
    if not COLLECT_DATA or not frames:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dish_slug = dish_name.lower().replace(" ", "_")[:20]
    step_num  = step.get("step_number", 0)
    uid       = uuid.uuid4().hex[:8]
    base_name = f"{dish_slug}_step{step_num}_{uid}"

    for i, frame in enumerate(frames):
        frame_to_jpeg_file(frame, str(DATA_DIR / f"{base_name}_f{i}.jpg"))

    label = {
        "dish_name":     dish_name,
        "step_number":   step_num,
        "action":        step.get("action", ""),
        "visual_cue":    step.get("visual_cue", ""),
        "trigger_type":  step.get("trigger_type", ""),
        "camera":        step.get("camera", ""),
        "n_frames":      len(frames),
        "vision_result": result,
        "frame_meta":    frame_meta,
        "collected_at":  datetime.now().isoformat(),
    }

    with open(DATA_DIR / f"{base_name}.json", "w") as f:
        json.dump(label, f, indent=2)

    print(f"  [camera] Saved {len(frames)} training frame(s) → {base_name}")
