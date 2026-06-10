"""
voice.py
--------
All spoken output for the cooking assistant.
Uses macOS `say` command — no API, no internet, works offline.

Voices:
  - Samantha (default) — clear, natural US English
  - You can change VOICE to any installed macOS voice

Usage:
  from voice import speak, speak_step, speak_alert, speak_teaching
"""

import subprocess
import threading


VOICE = "Samantha"
RATE  = 175  # words per minute (default ~180, lower = clearer)


# ── core speak ────────────────────────────────────────────────────────────────

def speak(text: str, block: bool = True):
    """
    Speak text aloud using macOS say.
    
    block=True  — wait until speech finishes before returning
    block=False — fire and forget (non-blocking)
    """
    if not text or not text.strip():
        return

    cmd = ["say", "-v", VOICE, "-r", str(RATE), text]

    if block:
        subprocess.run(cmd, capture_output=True)
    else:
        threading.Thread(target=subprocess.run, args=(cmd,),
                         kwargs={"capture_output": True}, daemon=True).start()


# ── structured speak helpers ──────────────────────────────────────────────────

def speak_step_intro(step: dict):
    """
    Read a step aloud when it begins.
    Includes the action + teaching explanation of why it matters.
    """
    action      = step.get("action", "")
    teaching    = step.get("teaching", {})
    why         = teaching.get("why", "")
    watch_for   = teaching.get("watch_for", "")
    trigger     = step.get("trigger_type", "instruction")
    duration    = step.get("duration_string") or (
        f"{int(step['duration_minutes'])} minutes"
        if step.get("duration_minutes") else None
    )
    temperature = step.get("temperature")

    parts = []

    # step number + action
    parts.append(f"Step {step['step_number']}. {action}.")

    # add duration / temperature context
    if duration and temperature:
        parts.append(f"Cook for {duration} at {temperature}.")
    elif duration:
        parts.append(f"This will take about {duration}.")
    elif temperature:
        parts.append(f"Use {temperature}.")

    # teaching: why this matters
    if why:
        parts.append(why)

    # for condition steps, tell the user what to watch for
    if trigger == "condition" and watch_for:
        parts.append(f"Watch for: {watch_for}.")
        interval = step.get("poll_interval_seconds", 30)
        parts.append(
            f"I'll check on this every {interval} seconds and let you know when it's ready."
        )

    # for timed steps, confirm the timer is running
    if trigger == "timed" and duration:
        parts.append(f"I've started a {duration} timer for you.")

    speak(" ".join(parts))


def speak_timer_alert(step: dict):
    """Alert fired when a timed step's countdown reaches zero."""
    action = step.get("action", "this step")
    speak(
        f"Time's up! Your {action} should be done. "
        f"Let me take a quick look — hold the camera over it for a moment."
    )


def speak_condition_check(step: dict):
    """
    Said before each periodic camera capture on a condition step.
    Tells the user the system is looking, and what it's checking for.
    """
    visual_cue = step.get("visual_cue") or step.get("teaching", {}).get("watch_for", "")
    action = step.get("action", "")
    speak(
        f"Checking on your {action}."
        + (f" Looking for: {visual_cue}." if visual_cue else ""),
        block=False,
    )


def speak_condition_not_ready(step: dict, observation: str):
    """
    Called when the vision model says the condition isn't met yet.
    Gives teaching feedback, not just 'not done'.
    """
    interval = step.get("poll_interval_seconds", 30)
    speak(
        f"{observation} Keep going — I'll check again in {interval} seconds.",
        block=False,
    )


def speak_condition_ready(step: dict, observation: str):
    """Called when the vision model confirms the condition is met."""
    speak(
        f"Looking good! {observation} "
        f"That's exactly what we want. Moving to the next step."
    )


def speak_common_mistake(step: dict):
    """
    Proactively warn about the most common mistake for this step.
    Called partway through timed steps or condition steps.
    """
    mistake = step.get("teaching", {}).get("common_mistake", "")
    if mistake:
        speak(f"Quick tip: {mistake}", block=False)


def speak_alert(message: str):
    """Generic urgent alert — heat, timing, safety."""
    speak(message)


def speak_recipe_complete(dish_name: str):
    """Final message when all steps are done."""
    speak(
        f"You've finished cooking {dish_name}! "
        f"Great work. Let it rest for a moment before serving."
    )


def speak_welcome(recipe: dict):
    """Opening message when a cooking session starts."""
    dish     = recipe.get("dish_name", "your dish")
    steps    = len(recipe.get("steps", []))
    cuisine  = recipe.get("cuisine")
    servings = recipe.get("servings")

    parts = [f"Let's cook {dish}."]
    if cuisine:
        parts.append(f"This is a {cuisine} dish.")
    if servings:
        parts.append(f"This recipe serves {servings}.")
    parts.append(
        f"There are {steps} steps. I'll guide you through each one, "
        f"tell you what to look for, and check on your progress automatically. "
        f"Let's get started."
    )
    speak(" ".join(parts))
