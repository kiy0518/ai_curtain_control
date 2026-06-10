"""Curtain controller — placeholder state machine.

The real motor controller (UART serial) is deferred to Phase M. For now this
records the desired curtain state from gestures or dashboard buttons so the UI
can reflect it. When the motor layer lands, ``_drive()`` will send serial
commands instead of just updating ``state``.
"""

import threading
import time


class CurtainController:
    # placeholder states
    OPEN, CLOSED, STOPPED, UNKNOWN = "OPEN", "CLOSED", "STOPPED", "UNKNOWN"

    def __init__(self, cooldown=2.0):
        self.state = self.UNKNOWN
        self.last_action = None
        self.last_source = None
        self.last_ts = 0.0
        self.motor_connected = False        # Phase M wires this to serial
        self._cooldown = cooldown
        self._lock = threading.Lock()

    def command(self, action, source, now=None):
        """action: OPEN/CLOSE/STOP. source: 'gesture'|'dashboard'|'schedule'.

        Returns True if accepted (not in cooldown / state change)."""
        now = now if now is not None else time.time()
        action = action.upper()
        if action not in ("OPEN", "CLOSE", "STOP"):
            return False
        with self._lock:
            # debounce repeated identical commands within cooldown
            if action == self.last_action and (now - self.last_ts) < self._cooldown:
                return False
            self.last_action = action
            self.last_source = source
            self.last_ts = now
            self._drive(action)
        return True

    def _drive(self, action):
        # Placeholder: just reflect intent. Phase M -> send $OPEN/$CLOSE/$STOP.
        if action == "OPEN":
            self.state = self.OPEN
        elif action == "CLOSE":
            self.state = self.CLOSED
        elif action == "STOP":
            self.state = self.STOPPED

    def snapshot(self):
        with self._lock:
            return {
                "state": self.state,
                "last_action": self.last_action,
                "last_source": self.last_source,
                "motor_connected": self.motor_connected,
            }
