"""
cook.py
-------
Entry point to start a cooking session.

Usage:
    python cook.py recipe_output.json
    python cook.py ../recipe_engine/recipe_output.json
"""

import sys
from pathlib import Path
from session import CookingSession


def main():
    if len(sys.argv) < 2:
        print("Usage: python cook.py <path_to_recipe_output.json>")
        print("Example: python cook.py ../recipe_engine/recipe_output.json")
        sys.exit(1)

    recipe_path = sys.argv[1]

    if not Path(recipe_path).exists():
        print(f"Error: recipe file not found: {recipe_path}")
        sys.exit(1)

    print(f"\nLoading recipe from: {recipe_path}")
    print("Press Ctrl+C at any time to stop the session.\n")

    session = CookingSession(recipe_path)

    try:
        session.start()
    except KeyboardInterrupt:
        print("\n\nSession interrupted.")
        session.stop()


if __name__ == "__main__":
    main()
