"""Schedule runner — fires curtain commands at time / sunrise / sunset.

Background thread; every ~20s it checks enabled schedules against the board's
local time and triggers ``controller.command(action, 'schedule')`` once per
occurrence. No external scheduler dependency.
"""

import threading
import time

import store
from solar import sun_times


def _applies_today(days_csv, weekday):
    if not days_csv:
        return True                       # "" = every day
    try:
        return weekday in {int(x) for x in days_csv.split(",") if x != ""}
    except ValueError:
        return True


def _target_minutes(sched, now):
    """Resolve a schedule to local minutes-of-day for `now`'s date, or None."""
    if sched["kind"] == "time":
        if sched["hh"] is None or sched["mm"] is None:
            return None
        return sched["hh"] * 60 + sched["mm"]
    # kind == 'sun'
    try:
        lat = float(store.get_setting("lat", "37.5665"))
        lon = float(store.get_setting("lon", "126.9780"))
    except (TypeError, ValueError):
        return None
    sr, ss = sun_times(lat, lon, now.tm_year, now.tm_mon, now.tm_mday,
                       now.tm_gmtoff or 0)
    base = sr if sched["sun_event"] == "sunrise" else ss
    if base is None:
        return None
    return int(round(base + (sched["sun_offset"] or 0)))


class SchedulerThread(threading.Thread):
    def __init__(self, controller, interval=20):
        super().__init__(daemon=True)
        self.controller = controller
        self.interval = interval
        self._fired = set()
        self._running = True

    def run(self):
        while self._running:
            try:
                self._tick()
            except Exception as e:        # never let the loop die
                print("scheduler error:", e)
            time.sleep(self.interval)

    def _tick(self):
        now = time.localtime()
        cur = now.tm_hour * 60 + now.tm_min
        datekey = (now.tm_year, now.tm_mon, now.tm_mday)
        # prune fired keys from previous days
        self._fired = {k for k in self._fired if k[0] == datekey}

        for s in store.list_schedules():
            if not s["enabled"] or not _applies_today(s["days"], now.tm_wday):
                continue
            tgt = _target_minutes(s, now)
            if tgt is None or tgt != cur:
                continue
            key = (datekey, s["id"], tgt)
            if key in self._fired:
                continue
            self._fired.add(key)
            self.controller.command(s["action"], "schedule")
            print(f"[scheduler] fired #{s['id']} {s['action']} ({s['name']})")

    def stop(self):
        self._running = False
