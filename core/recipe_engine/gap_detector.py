"""
gap_detector.py
---------------
Fills in missing information (duration, temperature, technique) by
searching the web for canonical recipe knowledge.

Strategy:
  1. Identify steps with gaps from the normalizer output.
  2. Build targeted search queries per gap type.
  3. Search using DuckDuckGo (free, no API key needed).
  4. Parse top results and use LLM to extract the specific missing value.
  5. Merge back into the recipe dict.

No API keys required — uses DuckDuckGo HTML search + requests.
"""

import re
import json
import time
import requests
import ollama
from urllib.parse import quote_plus


# ── DuckDuckGo search (no API key) ────────────────────────────────────────────

DDG_URL = "https://html.duckduckgo.com/html/"
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search DuckDuckGo and return list of {title, snippet, url} dicts.
    Rate-limited: adds a short sleep to avoid blocks.
    """
    time.sleep(0.8)  # polite delay
    try:
        resp = requests.post(
            DDG_URL,
            data={"q": query, "b": "", "kl": "us-en"},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [gap_detector] Search failed: {e}")
        return []

    # Parse results from HTML (DDG HTML endpoint)
    results = []
    # Find result blocks: <a class="result__a" href="...">title</a>
    # and <a class="result__snippet">snippet</a>
    title_pattern = re.compile(
        r'class="result__a"[^>]*>(.*?)</a>', re.S
    )
    snippet_pattern = re.compile(
        r'class="result__snippet"[^>]*>(.*?)</a>', re.S
    )
    url_pattern = re.compile(
        r'class="result__url"[^>]*>(.*?)</span>', re.S
    )

    titles   = [re.sub(r"<[^>]+>", "", t) for t in title_pattern.findall(resp.text)]
    snippets = [re.sub(r"<[^>]+>", "", s) for s in snippet_pattern.findall(resp.text)]
    urls     = [u.strip() for u in url_pattern.findall(resp.text)]

    for i in range(min(max_results, len(titles))):
        results.append({
            "title":   titles[i] if i < len(titles) else "",
            "snippet": snippets[i] if i < len(snippets) else "",
            "url":     urls[i] if i < len(urls) else "",
        })

    return results


# ── query builders ────────────────────────────────────────────────────────────

def _build_query(dish_name: str, step_action: str, gap_type: str) -> str:
    """Build a targeted search query for a specific gap."""
    action_clean = re.sub(r"\b(the|a|an|some)\b", "", step_action, flags=re.I).strip()

    if gap_type == "duration":
        return f"how long to {action_clean} {dish_name} minutes"
    elif gap_type == "temperature":
        return f"what temperature {action_clean} {dish_name} heat level"
    elif gap_type == "visual_cue":
        return f"what does {action_clean} {dish_name} look like when done"
    elif gap_type == "technique_note":
        return f"how to {action_clean} properly technique tips"
    else:
        return f"{action_clean} {dish_name} cooking guide"


# ── LLM extraction from search results ───────────────────────────────────────

EXTRACT_PROMPT = """You are a cooking expert. Based on the search results below,
extract the specific value requested. Be concise and precise.

Dish: {dish_name}
Step: {step_action}
Looking for: {gap_type}

Search results:
{search_text}

Extract ONLY the {gap_type} value. Examples:
- duration → "8-10 minutes" or "15 minutes"  
- temperature → "medium heat" or "180°C / 350°F"
- visual_cue → "until golden brown and edges start to crisp"
- technique_note → "cut parallel to the grain in 1cm slices"

If you cannot determine the value from the results, respond with: null

Respond with only the value, no explanation.
"""

def _extract_value_from_results(
    dish_name: str,
    step_action: str,
    gap_type: str,
    results: list[dict],
    model: str = "mistral",
) -> str | None:
    """Use LLM to extract a specific value from search result snippets."""
    if not results:
        return None

    search_text = "\n\n".join(
        f"[{r['title']}]\n{r['snippet']}" for r in results
    )

    prompt = EXTRACT_PROMPT.format(
        dish_name=dish_name,
        step_action=step_action,
        gap_type=gap_type,
        search_text=search_text[:3000],  # stay within context
    )

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    value = response["message"]["content"].strip()

    if value.lower() in ("null", "none", "", "unknown"):
        return None

    # strip quotes if LLM wrapped it
    value = value.strip('"\'')
    return value


# ── gap → recipe field mapping ────────────────────────────────────────────────

GAP_TO_FIELD = {
    "duration":       "duration_minutes",  # note: we store as string here
    "temperature":    "temperature",
    "visual_cue":     "visual_cue",
    "technique_note": "technique_note",
}

# For duration we need to parse minutes out of the string
def _parse_duration(value: str) -> float | None:
    """Convert '8-10 minutes' → 9.0 (midpoint). Returns None if unparseable."""
    if not value:
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", value)
    if not nums:
        return None
    floats = [float(n) for n in nums]
    # convert hours to minutes if needed
    if "hour" in value.lower():
        floats = [f * 60 for f in floats]
    return sum(floats) / len(floats)  # midpoint of range


# ── public API ────────────────────────────────────────────────────────────────

def fill_gaps(recipe: dict, model: str = "mistral") -> dict:
    """
    Takes a normalized recipe dict (from normalizer.py).
    Searches the web to fill gaps, returns updated recipe dict.
    
    Modifies recipe in-place and also returns it.
    """
    dish_name = recipe.get("dish_name", "this dish")
    steps = recipe.get("steps", [])

    total_gaps = sum(len(s.get("gaps", [])) for s in steps)
    if total_gaps == 0:
        print("[gap_detector] No gaps to fill ✓")
        return recipe

    print(f"[gap_detector] Filling {total_gaps} gaps across {len(steps)} steps…")

    for step in steps:
        gaps = step.get("gaps", [])
        if not gaps:
            continue

        action = step.get("action", "cook")
        print(f"  Step {step['step_number']}: '{action}' — gaps: {gaps}")

        filled = []
        for gap in gaps:
            query = _build_query(dish_name, action, gap)
            print(f"    Searching: '{query}'")
            results = _ddg_search(query)

            if not results:
                print(f"    No results for {gap}")
                continue

            value = _extract_value_from_results(
                dish_name, action, gap, results, model
            )

            if value is None:
                print(f"    Could not determine {gap}")
                continue

            # write into the step
            field = GAP_TO_FIELD.get(gap, gap)
            if gap == "duration":
                # store both human string and numeric minutes
                step["duration_string"] = value
                step["duration_minutes"] = _parse_duration(value)
            else:
                step[field] = value

            print(f"    ✓ {gap} = '{value}'")
            filled.append(gap)

        # remove filled gaps from the gaps list
        step["gaps"] = [g for g in gaps if g not in filled]

    remaining = sum(len(s.get("gaps", [])) for s in steps)
    print(f"[gap_detector] Done. {total_gaps - remaining}/{total_gaps} gaps filled ✓")
    return recipe
