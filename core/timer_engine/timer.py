"""
timer.py
--------
Manages timed countdowns and condition-watch polling.
All timers run in background threads so the main session loop stays responsive.

Two timer types:
  CountdownTimer  — fires a callback after N minutes (timed steps)
  ConditionPoller — fires a callback every N seconds until told to stop (condition steps)
"""

import threading
import time
from typing import Callable


# ── Countdown Timer ───────────────────────────────────────────────────────────

class CountdownTimer:
    """
    Counts down for `duration_seconds`, then calls `on_complete`.
    Optionally calls `on_halfway` at the midpoint (good for common-mistake tips).
    Can be cancelled before it fires.

    Usage:
        timer = CountdownTimer(
            duration_seconds=600,
            on_complete=lambda: speak("Time's up!"),
            on_halfway=lambda: speak("5 minutes left"),
        )
        timer.start()
        # ...
        timer.cancel()  # if you need to stop it early
    """

    def __init__(
        self,
        duration_seconds: float,
        on_complete: Callable,
        on_halfway: Callable | None = None,
        label: str = "timer",
    ):
        self.duration_seconds = duration_seconds
        self.on_complete      = on_complete
        self.on_halfway       = on_halfway
        self.label            = label
        self._cancelled       = threading.Event()
        self._thread          = None
        self.start_time       = None

    def start(self):
        self.start_time = time.time()
        self._cancelled.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"  [timer] ⏱  '{self.label}' started — {self.duration_seconds:.0f}s")

    def _run(self):
        halfway = self.duration_seconds / 2
        halfway_fired = False

        while True:
            elapsed = time.time() - self.start_time

            # halfway callback
            if self.on_halfway and not halfway_fired and elapsed >= halfway:
                halfway_fired = True
                try:
                    self.on_halfway()
                except Exception as e:
                    print(f"  [timer] halfway callback error: {e}")

            # done
            if elapsed >= self.duration_seconds:
                if not self._cancelled.is_set():
                    print(f"  [timer] ✓ '{self.label}' complete")
                    try:
                        self.on_complete()
                    except Exception as e:
                        print(f"  [timer] complete callback error: {e}")
                return

            # cancelled
            if self._cancelled.is_set():
                print(f"  [timer] '{self.label}' cancelled")
                return

            time.sleep(0.5)

    def cancel(self):
        self._cancelled.set()

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0
        return time.time() - self.start_time

    @property
    def remaining_seconds(self) -> float:
        return max(0, self.duration_seconds - self.elapsed_seconds)

    def __repr__(self):
        return (
            f"CountdownTimer(label={self.label!r}, "
            f"remaining={self.remaining_seconds:.0f}s)"
        )


# ── Condition Poller ──────────────────────────────────────────────────────────

class ConditionPoller:
    """
    Calls `on_poll` every `interval_seconds` until `stop()` is called
    or `on_poll` returns True (meaning condition is met).

    The `on_poll` callback should:
      - Capture a camera frame
      - Run vision check
      - Return True if condition is met, False to keep polling

    Usage:
        poller = ConditionPoller(
            interval_seconds=30,
            on_poll=check_onions_translucent,
            label="onion check",
        )
        poller.start()
        # poller stops itself when on_poll returns True
        # or call poller.stop() to cancel
    """

    def __init__(
        self,
        interval_seconds: float,
        on_poll: Callable[[], bool],
        on_timeout: Callable | None = None,
        max_polls: int = 20,
        label: str = "condition",
    ):
        self.interval_seconds = interval_seconds
        self.on_poll          = on_poll
        self.on_timeout       = on_timeout
        self.max_polls        = max_polls
        self.label            = label
        self._stop_event      = threading.Event()
        self._thread          = None
        self.poll_count       = 0

    def start(self):
        self._stop_event.clear()
        self.poll_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"  [poller] 👁  '{self.label}' polling every {self.interval_seconds}s")

    def _run(self):
        # first poll after one interval (give user time to start)
        self._stop_event.wait(timeout=self.interval_seconds)

        while not self._stop_event.is_set():
            self.poll_count += 1
            print(f"  [poller] '{self.label}' — poll #{self.poll_count}")

            try:
                condition_met = self.on_poll()
            except Exception as e:
                print(f"  [poller] poll error: {e}")
                condition_met = False

            if condition_met:
                print(f"  [poller] ✓ '{self.label}' condition met after {self.poll_count} polls")
                return

            if self.poll_count >= self.max_polls:
                print(f"  [poller] '{self.label}' max polls reached")
                if self.on_timeout:
                    try:
                        self.on_timeout()
                    except Exception as e:
                        print(f"  [poller] timeout callback error: {e}")
                return

            # wait for next interval (or stop signal)
            self._stop_event.wait(timeout=self.interval_seconds)

    def stop(self):
        self._stop_event.set()
        print(f"  [poller] '{self.label}' stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ── Simple one-shot delay ─────────────────────────────────────────────────────

def after(seconds: float, callback: Callable):
    """Fire callback once after a delay. Non-blocking."""
    def _run():
        time.sleep(seconds)
        try:
            callback()
        except Exception as e:
            print(f"  [after] callback error: {e}")

    threading.Thread(target=_run, daemon=True).start()
