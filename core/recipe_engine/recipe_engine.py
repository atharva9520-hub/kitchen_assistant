"""
recipe_engine.py
----------------
Main entry point. Chains:
  extractor → normalizer → gap_detector → classifier

Usage:
  from recipe_engine import build_recipe

  recipe = build_recipe("https://www.youtube.com/watch?v=...")
  recipe = build_recipe("https://www.instagram.com/reel/...")
  recipe = build_recipe("Chop onions, sauté for 10 minutes, add tomatoes...")

Output: fully structured recipe dict ready for the timer engine.
Saved to: recipe_output.json in current directory.
"""

import json
import time
from pathlib import Path

from extractor    import extract
from normalizer   import normalize
from gap_detector import fill_gaps
from classifier   import classify


def build_recipe(
    input_data: str,
    model: str = "mistral",
    skip_gap_fill: bool = False,
    output_path: str = "recipe_output.json",
) -> dict:
    """
    Full pipeline: input → structured, classified recipe.
    
    Args:
        input_data:    YouTube URL, Instagram URL, or plain text recipe
        model:         Ollama model name (default: mistral)
        skip_gap_fill: Set True to skip web search (faster, offline)
        output_path:   Where to save the JSON output
    
    Returns:
        Fully structured recipe dict
    """
    start = time.time()
    print("=" * 60)
    print("RECIPE ENGINE — starting pipeline")
    print("=" * 60)

    # 1. Extract
    print("\n[1/4] Extracting raw text…")
    extracted = extract(input_data)
    print(f"  Source: {extracted['source']}")
    print(f"  Title:  {extracted.get('title') or 'unknown'}")
    print(f"  Text length: {len(extracted['raw_text'])} chars")

    # 2. Normalize
    print("\n[2/4] Normalizing with LLM…")
    recipe = normalize(extracted, model=model)
    print(f"  Dish: {recipe.get('dish_name')}")
    print(f"  Steps: {len(recipe.get('steps', []))}")

    # 3. Gap fill
    if skip_gap_fill:
        print("\n[3/4] Skipping gap detection (skip_gap_fill=True)")
    else:
        print("\n[3/4] Filling gaps via web search…")
        recipe = fill_gaps(recipe, model=model)

    # 4. Classify
    print("\n[4/4] Classifying steps…")
    recipe = classify(recipe, model=model)

    # Save output
    out = Path(output_path)
    out.write_text(json.dumps(recipe, indent=2))
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {time.time() - start:.1f}s")
    print(f"Recipe saved → {out.resolve()}")
    print(f"{'=' * 60}\n")

    _print_summary(recipe)
    return recipe


def _print_summary(recipe: dict):
    """Print a human-readable summary of the processed recipe."""
    print(f"\n📋 {recipe.get('dish_name', 'Unknown dish').upper()}")
    print(f"   Cuisine: {recipe.get('cuisine') or 'not specified'}")
    print(f"   Serves:  {recipe.get('servings') or 'not specified'}")
    print(f"\n   Ingredients:")
    for ing in recipe.get("ingredients", []):
        print(f"     • {ing}")

    print(f"\n   Steps ({len(recipe.get('steps', []))}):")
    icons = {"timed": "⏱ ", "condition": "👁 ", "instruction": "▶ "}
    for step in recipe.get("steps", []):
        t = step.get("trigger_type", "instruction")
        icon = icons.get(t, "  ")
        dur = ""
        if step.get("duration_minutes"):
            dur = f" [{step['duration_minutes']:.0f} min]"
        elif step.get("duration_string"):
            dur = f" [{step['duration_string']}]"
        poll = ""
        if step.get("poll_interval_seconds"):
            poll = f" [check every {step['poll_interval_seconds']}s]"
        print(f"     {icon} {step['step_number']}. {step['action']}{dur}{poll}")
        if step.get("teaching", {}).get("why"):
            print(f"        → {step['teaching']['why'][:80]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python recipe_engine.py <youtube_url|instagram_url|'plain text'>")
        print("       python recipe_engine.py --skip-gaps <url>")
        sys.exit(1)

    skip = "--skip-gaps" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    input_val = args[0]

    build_recipe(input_val, skip_gap_fill=skip)
