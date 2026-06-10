"""BLE GATT peripheral — curtain remote control (Phase 7).

Advertises a Curtain Control service; the Flutter app (BLE central) writes
commands and subscribes to status notifications. Commands feed the shared
``CurtainController`` (alongside gesture / dashboard / schedule).

Security: relies on BLE bonding (pairing) at the OS/BlueZ level (decision:
bonding-only). Runs an asyncio loop in its own thread so it can live next to
the stdlib HTTP dashboard.

Library: bless (BlueZ peripheral). UUIDs are custom 128-bit (not SIG-based).
"""

import asyncio
import json
import threading

from bless import (BlessServer, GATTCharacteristicProperties,
                   GATTAttributePermissions)

SVC_UUID = "c0de0000-1212-efde-1523-785feabcd123"   # Curtain Control Service
CMD_UUID = "c0de0001-1212-efde-1523-785feabcd123"   # Command  (write)
STA_UUID = "c0de0002-1212-efde-1523-785feabcd123"   # Status   (read/notify)

DEVICE_NAME = "AI-Curtain"


class BleServerThread(threading.Thread):
    """Run the BLE peripheral in a background asyncio loop."""

    def __init__(self, controller, status_fn=None, name=DEVICE_NAME):
        super().__init__(daemon=True)
        self.controller = controller
        self.status_fn = status_fn or (lambda: controller.snapshot())
        self.name = name
        self.active = False
        self.error = None
        self._loop = None
        self._server = None
        self._stop = None

    # --- GATT callbacks ----------------------------------------------------
    def _on_read(self, characteristic, **kwargs):
        return self._status_bytes()

    def _on_write(self, characteristic, value, **kwargs):
        try:
            cmd = bytes(value).decode("utf-8", "ignore").strip().upper()
        except Exception:
            return
        if cmd in ("OPEN", "CLOSE", "STOP"):
            self.controller.command(cmd, "ble")
            self._push_status()

    def _status_bytes(self):
        snap = self.status_fn() or {}
        return json.dumps({"state": snap.get("state"),
                           "motor": snap.get("motor_connected", False)}).encode()

    def _push_status(self):
        if self._server is not None:
            try:
                ch = self._server.get_characteristic(STA_UUID)
                ch.value = self._status_bytes()
                self._server.update_value(SVC_UUID, STA_UUID)
            except Exception:
                pass

    # --- thread ------------------------------------------------------------
    def run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:
            self.error = str(e)
            self.active = False

    async def _main(self):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        server = BlessServer(name=self.name, loop=self._loop)
        server.read_request_func = self._on_read
        server.write_request_func = self._on_write

        await server.add_new_service(SVC_UUID)
        await server.add_new_characteristic(
            SVC_UUID, CMD_UUID,
            (GATTCharacteristicProperties.write |
             GATTCharacteristicProperties.write_without_response),
            None, GATTAttributePermissions.writeable)
        await server.add_new_characteristic(
            SVC_UUID, STA_UUID,
            (GATTCharacteristicProperties.read |
             GATTCharacteristicProperties.notify),
            self._status_bytes(), GATTAttributePermissions.readable)

        await server.start()
        self._server = server
        self.active = True

        # periodic status notify (reflects gesture/schedule-driven changes too)
        last = None
        try:
            while not self._stop.is_set():
                cur = self._status_bytes()
                if cur != last:
                    self._push_status()
                    last = cur
                await asyncio.sleep(1.0)
        finally:
            await server.stop()
            self.active = False

    def stop(self):
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)


# standalone smoke test: advertise for ~20s with a dummy controller
if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, ".")

    class _Dummy:
        def __init__(self):
            self.state = "UNKNOWN"
        def command(self, action, source):
            self.state = {"OPEN": "OPEN", "CLOSE": "CLOSED", "STOP": "STOPPED"}[action]
            print(f"[ble] command {action} from {source} -> {self.state}")
            return True
        def snapshot(self):
            return {"state": self.state, "motor_connected": False}

    t = BleServerThread(_Dummy())
    t.start()
    for _ in range(20):
        time.sleep(1)
        if t.error:
            print("BLE 오류:", t.error); break
        if t.active:
            print("BLE 광고중… (nRF Connect/폰으로 'AI-Curtain' 검색)")
            break
    time.sleep(18)
    t.stop()
