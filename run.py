"""
run.py
------
Main entry point for kitchen_assistant.

Selects a profile (cooking or cocktail), injects it into sys.path so all
core modules automatically import the profile's overrides, then runs the
recipe engine followed by the cooking session.

Usage:
    python run.py --mode cook    <youtube_url | instagram_url | "plain text">
    python run.py --mode bartend <youtube_url | instagram_url | "plain text">

    # Skip recipe building (use existing recipe_output.json):
    python run.py --mode cook    --recipe recipe_output.json
    python run.py --mode bartend --recipe recipe_output.json

    # Skip web gap-filling (faster, offline):
    python run.py --mode cook --skip-gaps <url>

Examples:
    python run.py --mode cook "https://youtube.com/watch?v=abc123"
    python run.py --mode bartend "https://instagram.com/reel/xyz"
    python run.py --mode cook --recipe outputs/lasagne.json
"""

import sys
import os
import argparse
from pathlib import Path


# ── path setup ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.resolve()


def setup_paths(mode: str):
    """
    Inject paths into sys.path so imports resolve correctly.

    Order matters:
      1. Profile folder  — profile overrides (classifier, prompts, model_router)
      2. Core engines    — shared modules (session, timer, voice, etc.)
      3. Root            — for run.py itself

    Python finds the profile's version of any module before the core base,
    which is exactly how the override mechanism works.
    """
    mode_to_profile = {"cook": "cooking", "bartend": "cocktail"}
    profile_name = mode_to_profile.get(mode, mode)
    profile_dir  = ROOT / "profiles" / profile_name

    if not profile_dir.exists():
        print(f"Error: profile '{mode}' not found at {profile_dir}")
        sys.exit(1)

    paths_to_add = [
        str(profile_dir),                        # profile overrides first
        str(ROOT / "core" / "vision_engine"),    # vision base
        str(ROOT / "core" / "timer_engine"),     # session, timer, voice
        str(ROOT / "core" / "recipe_engine"),    # extractor, normalizer, etc.
        str(ROOT / "core"),                      # fallback
        str(ROOT),                               # root
    ]

    for p in reversed(paths_to_add):
        if p not in sys.path:
            sys.path.insert(0, p)

    print(f"[run] Mode: {mode}")
    print(f"[run] Profile: {profile_dir}")


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Kitchen Assistant — cooking and cocktail teacher"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["cook", "bartend"],
        help="cook = cooking mode, bartend = cocktail mode",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="YouTube URL, Instagram URL, or plain text recipe",
    )
    parser.add_argument(
        "--recipe",
        help="Path to existing recipe_output.json (skip recipe building)",
    )
    parser.add_argument(
        "--skip-gaps",
        action="store_true",
        help="Skip web gap-filling (faster, works offline)",
    )
    parser.add_argument(
        "--output",
        default="recipe_output.json",
        help="Where to save the recipe JSON (default: recipe_output.json)",
    )
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1. inject profile into sys.path
    setup_paths(args.mode)

    # 2. now safe to import — profile overrides are in place
    from recipe_engine import build_recipe
    from session       import CookingSession

    # 3. build or load recipe
    if args.recipe:
        recipe_path = args.recipe
        if not Path(recipe_path).exists():
            print(f"Error: recipe file not found: {recipe_path}")
            sys.exit(1)
        print(f"[run] Using existing recipe: {recipe_path}")
    else:
        if not args.input:
            print("Error: provide a URL/text or use --recipe to load existing.")
            sys.exit(1)

        print(f"[run] Building recipe from: {args.input[:60]}")
        recipe = build_recipe(
            args.input,
            skip_gap_fill=args.skip_gaps,
            output_path=args.output,
        )
        recipe_path = args.output

    # 4. run session
    print(f"\n[run] Starting session → {recipe_path}\n")
    session = CookingSession(recipe_path)

    try:
        session.start()
    except KeyboardInterrupt:
        print("\n\nSession stopped.")
        session.stop()


if __name__ == "__main__":
    main()