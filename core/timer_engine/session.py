"""
session.py
----------
The main cooking session coordinator.
Loads a recipe, walks the user through every step, manages timers,
polls the camera for condition steps, and speaks guidance aloud.

Usage:
    from session import CookingSession

    session = CookingSession("recipe_output.json")
    session.start()
"""

import json
import time
import threading
from pathlib import Path

from voice  import (
    speak, speak_welcome, speak_step_intro, speak_timer_alert,
    speak_condition_check, speak_condition_not_ready,
    speak_condition_ready, speak_common_mistake, speak_recipe_complete,
)
from timer       import CountdownTimer, ConditionPoller
from vision      import check_condition
from voice_input import VoiceListener, check_dependencies


class CookingSession:
    """
    Runs a full cooking session from a classified recipe JSON file.

    Step flow:
      instruction → speak intro, wait for user to say 'next' or press Enter
      timed       → speak intro, start countdown, speak alert at end, vision check
      condition   → speak intro, start poller, speak feedback each poll, advance when done
    """

    def __init__(self, recipe_path: str):
        path = Path(recipe_path)
        if not path.exists():
            raise FileNotFoundError(f"Recipe not found: {recipe_path}")

        with open(path) as f:
            self.recipe = json.load(f)

        self.dish_name   = self.recipe.get("dish_name", "your dish")
        self.steps       = self.recipe.get("steps", [])
        self.current_idx = 0
        self.running     = False

        # active timer/poller — only one at a time
        self._active_timer:  CountdownTimer  | None = None
        self._active_poller: ConditionPoller | None = None

        # event to signal step completion from background threads
        self._step_done = threading.Event()

        # voice listener — started in start(), stopped in stop()
        self._voice_listener: VoiceListener | None = None
        self._last_spoken_step: dict | None = None  # for repeat command

        # check if mic is available
        deps = check_dependencies()
        self._voice_available = deps.get("microphone", False)
        if not self._voice_available:
            print(
                "\n[session] Microphone not available — using keyboard input.\n"
                "          To enable voice: brew install portaudio && pip install pyaudio\n"
            )

    # ── session control ───────────────────────────────────────────────────────

    def start(self):
        """Start the cooking session from the beginning."""
        if not self.steps:
            speak("This recipe has no steps. Please check the recipe file.")
            return

        self.running     = True
        self.current_idx = 0

        # start voice listener if mic available
        if self._voice_available:
            self._voice_listener = VoiceListener(on_command=self._handle_voice_command)
            self._voice_listener.start()

        speak_welcome(self.recipe)
        time.sleep(1)

        while self.running and self.current_idx < len(self.steps):
            step = self.steps[self.current_idx]
            self._run_step(step)
            self.current_idx += 1

        if self.running:
            speak_recipe_complete(self.dish_name)
            self.running = False

    def stop(self):
        """Stop the session early."""
        self.running = False
        if self._active_timer:
            self._active_timer.cancel()
        if self._active_poller:
            self._active_poller.stop()
        if self._voice_listener:
            self._voice_listener.stop()
        speak("Session paused. Come back when you're ready.")

    def skip_step(self):
        """Skip current step (called externally e.g. from voice command)."""
        speak("Skipping this step.")
        self._step_done.set()

    # ── voice command handler ─────────────────────────────────────────────────

    def _handle_voice_command(self, command: str):
        """
        Receives recognised voice commands from VoiceListener.
        Called from a background thread — all actions must be thread-safe.
        """
        if command == "next":
            speak("Moving on.", block=False)
            self._step_done.set()

        elif command == "skip":
            speak("Skipping this step.", block=False)
            self._step_done.set()

        elif command == "repeat":
            if self._last_spoken_step:
                speak("Here's that step again.", block=True)
                speak_step_intro(self._last_spoken_step)
            else:
                speak("Nothing to repeat yet.", block=False)

        elif command == "stop":
            speak("Pausing the session. Say 'next' or press Enter to resume.", block=False)
            # don't set _step_done — just pause voice listener
            if self._voice_listener:
                self._voice_listener.mute()
            input("  [paused] Press Enter to resume…")
            if self._voice_listener:
                self._voice_listener.unmute()
            speak("Resuming.", block=False)

        elif command == "timer":
            if self._active_timer:
                remaining = self._active_timer.remaining_seconds
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                if mins > 0:
                    speak(f"{mins} minutes and {secs} seconds remaining.", block=False)
                else:
                    speak(f"{secs} seconds remaining.", block=False)
            else:
                speak("No active timer right now.", block=False)

        elif command == "help":
            speak(
                "You can say: next or done to move on, "
                "skip to skip this step, "
                "repeat to hear the step again, "
                "timer to hear how much time is left, "
                "or stop to pause.",
                block=False,
            )

    # ── step runners ──────────────────────────────────────────────────────────

    def _run_step(self, step: dict):
        """Dispatch to the right runner based on trigger_type."""
        trigger = step.get("trigger_type", "instruction")
        self._step_done.clear()

        print(f"\n{'─' * 50}")
        print(f"Step {step['step_number']}/{len(self.steps)} [{trigger}]: {step['action']}")
        print(f"{'─' * 50}")

        # mute mic while we're speaking to avoid feedback
        if self._voice_listener:
            self._voice_listener.mute()

        self._last_spoken_step = step
        speak_step_intro(step)

        if self._voice_listener:
            self._voice_listener.unmute()

        if trigger == "timed":
            self._run_timed_step(step)
        elif trigger == "condition":
            self._run_condition_step(step)
        else:
            self._run_instruction_step(step)

    def _run_instruction_step(self, step: dict):
        """
        Instruction step: read aloud, then wait for user to confirm ready.
        Uses a simple Enter keypress for now (voice command hook ready).
        """
        self._wait_for_user(step)

    def _run_timed_step(self, step: dict):
        """
        Timed step:
          - Start countdown
          - At halfway: speak common mistake tip
          - At end: speak alert, do a vision check
          - Then wait for user to confirm before advancing
        """
        duration_min = step.get("duration_minutes")
        if not duration_min:
            # no duration — treat as instruction
            self._run_instruction_step(step)
            return

        duration_sec = duration_min * 60

        def on_halfway():
            speak_common_mistake(step)

        def on_complete():
            speak_timer_alert(step)
            time.sleep(2)
            # vision check after timer
            self._do_vision_check(step)
            self._step_done.set()

        self._active_timer = CountdownTimer(
            duration_seconds=duration_sec,
            on_complete=on_complete,
            on_halfway=on_halfway,
            label=step.get("action", "step"),
        )
        self._active_timer.start()

        # wait for the timer to finish (or user to skip)
        self._step_done.wait()
        self._active_timer = None

    def _run_condition_step(self, step: dict):
        """
        Condition step:
          - Poll camera every N seconds
          - Speak teaching feedback each poll
          - Stop when vision model says condition is met
        """
        interval = step.get("poll_interval_seconds", 30)

        def on_poll() -> bool:
            """Called every interval seconds. Returns True when condition met."""
            speak_condition_check(step)
            time.sleep(2)  # brief pause so speech finishes before capture

            result = check_condition(step, self.dish_name)
            feedback = result.get("feedback", "")
            done     = result.get("done", False)

            if done:
                speak_condition_ready(step, feedback)
                self._step_done.set()
                return True
            else:
                speak_condition_not_ready(step, feedback)
                return False

        def on_timeout():
            speak(
                f"I've been checking for a while — your {step.get('action', 'food')} "
                f"is taking longer than expected. Keep going and let me know "
                f"when you think it's ready."
            )
            self._step_done.set()

        self._active_poller = ConditionPoller(
            interval_seconds=interval,
            on_poll=on_poll,
            on_timeout=on_timeout,
            max_polls=20,
            label=step.get("action", "condition"),
        )
        self._active_poller.start()

        # wait for condition to be met (or timeout / skip)
        self._step_done.wait()
        self._active_poller.stop()
        self._active_poller = None

    # ── vision check ─────────────────────────────────────────────────────────

    def _do_vision_check(self, step: dict):
        """Run a one-off vision check (used after timed steps complete)."""
        speak("Let me take a quick look.", block=True)
        time.sleep(1)
        result = check_condition(step, self.dish_name)
        feedback = result.get("feedback", "")
        done     = result.get("done", False)

        if done:
            speak(f"Looks perfect! {feedback}")
        else:
            speak(
                f"{feedback} "
                f"You may want to give it a bit more time, but let's move on."
            )

    # ── user input ────────────────────────────────────────────────────────────

    def _wait_for_user(self, step: dict):
        """
        Wait for user to confirm ready for next step.
        If mic available: listening for 'next' / 'done' / 'skip'.
        Fallback: press Enter.
        """
        action = step.get("action", "this step")

        if self._voice_available:
            print(f"\n  ▶  Say 'next' or 'done' when you've completed: {action}")
            print(f"     (or press Enter, type 's' to skip)\n")
        else:
            print(f"\n  ▶  Press Enter when you've completed: {action}")
            print(f"     (or type 's' + Enter to skip)\n")

        # wait for either voice command (_step_done set by handler)
        # or keyboard input
        while not self._step_done.is_set():
            try:
                # non-blocking check every 0.2s
                import select, sys
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if ready:
                    user_input = sys.stdin.readline().strip().lower()
                    if user_input == "s":
                        speak("Moving on.")
                    self._step_done.set()
            except Exception:
                # select not available (Windows) — blocking fallback
                user_input = input("  > ").strip().lower()
                if user_input == "s":
                    speak("Moving on.")
                self._step_done.set()