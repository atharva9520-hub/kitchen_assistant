"""
extractor.py
------------
Pulls raw text from YouTube URLs, Instagram reel URLs, or plain text.

YouTube  : yt-dlp fetches auto-captions (preferred) or description.
           If neither exists, downloads audio and runs Whisper locally.
Instagram: yt-dlp fetches caption + audio transcript via Whisper.
Plain text: returned as-is.

Output is always a dict:
{
    "source":      "youtube" | "instagram" | "text",
    "title":       str | None,
    "raw_text":    str,          # transcript / caption / plain text
    "description": str | None,  # video description if available
}
"""

import os
import re
import json
import tempfile
import subprocess
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _is_youtube(url: str) -> bool:
    return bool(re.search(r"(youtube\.com|youtu\.be)", url))

def _is_instagram(url: str) -> bool:
    return bool(re.search(r"instagram\.com", url))


def _run_yt_dlp(url: str, extra_args: list[str]) -> dict:
    """Run yt-dlp and return parsed JSON info dict."""
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-playlist",
        *extra_args,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:300]}")
    return json.loads(result.stdout)


def _captions_from_info(info: dict) -> str | None:
    """
    Try to get English auto-captions from yt-dlp info dict.
    Returns plain text or None.
    """
    subtitles = info.get("automatic_captions") or info.get("subtitles") or {}
    # prefer English
    for lang in ("en", "en-US", "en-GB"):
        if lang in subtitles:
            entries = subtitles[lang]
            # look for a json3 or vtt format we can parse inline
            for entry in entries:
                if entry.get("ext") == "json3":
                    return _fetch_json3_captions(entry["url"])
    return None


def _fetch_json3_captions(url: str) -> str:
    """Download json3 caption file and extract plain text."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())
    lines = []
    for event in data.get("events", []):
        for seg in event.get("segs", []):
            t = seg.get("utf8", "").strip()
            if t and t != "\n":
                lines.append(t)
    return " ".join(lines)


def _whisper_transcribe(audio_path: str) -> str:
    """Run OpenAI Whisper locally on an audio file. Returns transcript text."""
    import whisper  # loaded lazily — slow import
    print("  [whisper] Loading model (base.en)…")
    model = whisper.load_model("base.en")
    print("  [whisper] Transcribing…")
    result = model.transcribe(audio_path, language="en")
    return result["text"]


def _download_audio(url: str, out_dir: str) -> str:
    """Download audio-only via yt-dlp into out_dir. Returns file path."""
    template = os.path.join(out_dir, "audio.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "mp3",
        "-o", template,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Audio download failed: {result.stderr[:300]}")
    # find the file
    for f in Path(out_dir).glob("audio.*"):
        return str(f)
    raise FileNotFoundError("Audio file not found after download")


# ── public API ───────────────────────────────────────────────────────────────

def extract_youtube(url: str) -> dict:
    """
    Extract recipe text from a YouTube video.
    Strategy:
      1. Try auto-captions (fast, no GPU needed)
      2. Fall back to video description
      3. Fall back to Whisper transcription (slower)
    """
    print(f"[extractor] YouTube: {url}")
    info = _run_yt_dlp(url, [])

    title = info.get("title")
    description = info.get("description", "")

    # 1. captions
    transcript = _captions_from_info(info)
    if transcript:
        print("  [extractor] Got captions ✓")
        return {
            "source": "youtube",
            "title": title,
            "raw_text": transcript,
            "description": description,
        }

    # 2. description (often has full recipe in food channels)
    if description and len(description) > 100:
        print("  [extractor] Using description as recipe text ✓")
        return {
            "source": "youtube",
            "title": title,
            "raw_text": description,
            "description": description,
        }

    # 3. whisper fallback
    print("  [extractor] No captions — downloading audio for Whisper…")
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = _download_audio(url, tmpdir)
        transcript = _whisper_transcribe(audio_path)

    return {
        "source": "youtube",
        "title": title,
        "raw_text": transcript,
        "description": description,
    }


def extract_instagram(url: str) -> dict:
    """
    Extract recipe text from an Instagram reel.
    Strategy:
      1. Pull caption (often has ingredients + brief steps)
      2. Always also Whisper-transcribe the audio (reels are mostly spoken)
      3. Combine both
    """
    print(f"[extractor] Instagram: {url}")
    info = _run_yt_dlp(url, [])

    title = info.get("title") or info.get("fulltitle")
    caption = info.get("description", "")  # yt-dlp puts caption here

    # always transcribe audio for reels — they're voice-driven
    print("  [extractor] Downloading audio for Whisper…")
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = _download_audio(url, tmpdir)
        transcript = _whisper_transcribe(audio_path)

    # combine: caption first (structured), then transcript (spoken detail)
    combined = ""
    if caption:
        combined += f"Caption:\n{caption}\n\n"
    combined += f"Spoken audio:\n{transcript}"

    return {
        "source": "instagram",
        "title": title,
        "raw_text": combined,
        "description": caption,
    }


def extract_text(text: str) -> dict:
    """Plain text — returned directly, no processing needed."""
    print("[extractor] Plain text input ✓")
    return {
        "source": "text",
        "title": None,
        "raw_text": text.strip(),
        "description": None,
    }


def extract(input_data: str) -> dict:
    """
    Main entry point. Accepts a URL or plain text string.
    Auto-detects the source type.
    """
    s = input_data.strip()
    if _is_youtube(s):
        return extract_youtube(s)
    elif _is_instagram(s):
        return extract_instagram(s)
    else:
        return extract_text(s)
