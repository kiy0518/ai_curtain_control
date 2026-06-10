"""Remote access via Cloudflare Tunnel (cloudflared) — unique public address.

Starts a *quick tunnel* (no account needed) that exposes the local dashboard at
a public ``https://<random>.trycloudflare.com`` URL. For a STABLE custom address
use a named tunnel (Cloudflare account + domain) — see work-plan/06.

Pure subprocess management; cloudflared binary resolved from ~/.local/bin or PATH.
"""

import os
import re
import shutil
import subprocess
import threading

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _find_cloudflared():
    local = os.path.expanduser("~/.local/bin/cloudflared")
    if os.path.exists(local):
        return local
    return shutil.which("cloudflared")


class RemoteManager:
    """Start/stop a cloudflared quick tunnel; expose the assigned public URL."""

    def __init__(self, port):
        self.port = port
        self.bin = _find_cloudflared()
        self.url = None
        self.active = False
        self.error = None
        self._proc = None
        self._lock = threading.Lock()

    def available(self):
        return self.bin is not None

    def start(self):
        with self._lock:
            if self.active:
                return True
            if not self.bin:
                self.error = "cloudflared 미설치 (~/.local/bin/cloudflared)"
                return False
            self.url = None
            self.error = None
            self._proc = subprocess.Popen(
                [self.bin, "tunnel", "--no-autoupdate",
                 "--url", f"http://localhost:{self.port}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            self.active = True
            threading.Thread(target=self._read, args=(self._proc,), daemon=True).start()
            return True

    def _read(self, proc):
        for line in proc.stdout:
            if self.url is None:
                m = _URL_RE.search(line)
                if m:
                    self.url = m.group(0)
        # process ended
        with self._lock:
            if proc is self._proc:
                self.active = False
                self.url = None

    def stop(self):
        with self._lock:
            if self._proc is not None:
                self._proc.terminate()
                self._proc = None
            self.active = False
            self.url = None

    def status(self):
        return {"available": self.available(), "active": self.active,
                "url": self.url, "error": self.error}
