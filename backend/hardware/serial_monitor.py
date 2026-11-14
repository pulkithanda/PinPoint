"""
Electrical check: read `dist=34 lcd=1` lines from the Arduino over USB serial.

Real readings that change when a hand moves prove the build works; silent or
stuck values point to a likely wrong port.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)

# USB vendor IDs: Arduino, Arduino (older), Seeed, WCH CH340 clones, FTDI, Silicon Labs
_KNOWN_VIDS = {0x2341, 0x2A03, 0x2886, 0x1A86, 0x0403, 0x10C4}


def parse_line(line: str) -> dict[str, int]:
    """'dist=34 lcd=1' -> {'dist': 34, 'lcd': 1}. Ignores anything malformed."""
    out: dict[str, int] = {}
    for tok in line.replace(",", " ").split():
        k, sep, v = tok.partition("=")
        if sep:
            try:
                out[k.strip().lower()] = int(float(v))
            except ValueError:
                pass
    return out


@dataclass
class SerialStatus:
    connected: bool          # serial port open
    receiving: bool          # lines arrived in the last 2 s
    last_dist: int | None
    dist_varies: bool        # distance changed by > 5 cm over the recent window
    sensor_dead: bool        # sensor returns 0 (timeout) consistently -> likely not on its port
    lcd_ok: bool | None      # None if the sketch doesn't report it
    seconds_receiving: float

    @property
    def works(self) -> bool:
        return self.receiving and not self.sensor_dead and self.lcd_ok is not False


def diagnose(st: SerialStatus, sensor_port: str = "D7", lcd_port: str = "an I 2 C port") -> str | None:
    """Plain-language problem description, or None if it looks healthy."""
    if not st.connected:
        return "I can't see the Arduino yet. Plug the USB cable into the Arduino."
    if not st.receiving:
        return "The Arduino is connected but not sending readings. Try unplugging the USB cable and plugging it back in."
    if st.sensor_dead:
        return f"I'm not getting distances from the sensor. Check the distance sensor is in {sensor_port}."
    if st.lcd_ok is False:
        return f"The screen isn't answering. Check the screen's cable is in {lcd_port}."
    return None


class SerialMonitor:
    def __init__(self, port: str = "auto", baud: int = 9600, window_s: float = 6.0):
        self.port = port
        self.baud = baud
        self.window_s = window_s
        self.samples: deque[tuple[float, dict[str, int]]] = deque(maxlen=400)
        self._ser = None
        self._first_rx: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="serial", daemon=True)
        self._lock = threading.Lock()

    # -- lifecycle --
    def start(self) -> "SerialMonitor":
        if self.port.lower() != "none":
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    @staticmethod
    def find_port() -> str | None:
        from serial.tools import list_ports

        ports = list(list_ports.comports())
        for p in ports:
            if p.vid in _KNOWN_VIDS:
                return p.device
        for p in ports:
            if "arduino" in (p.description or "").lower() or "usb" in (p.description or "").lower():
                return p.device
        return None

    def _run(self) -> None:
        import serial

        while not self._stop.is_set():
            device = self.port if self.port.lower() != "auto" else self.find_port()
            if not device:
                time.sleep(1.0)
                continue
            try:
                with serial.Serial(device, self.baud, timeout=0.5) as ser:
                    log.info("Serial connected on %s", device)
                    self._ser = ser
                    while not self._stop.is_set():
                        raw = ser.readline()
                        if raw:
                            self.feed(raw.decode("ascii", errors="ignore"))
            except (serial.SerialException, OSError) as e:
                log.info("Serial %s unavailable (%s); retrying", device, e)
            finally:
                self._ser = None
                self._first_rx = None
            time.sleep(1.0)

    # -- data --
    def feed(self, line: str, now: float | None = None) -> None:
        vals = parse_line(line)
        if "dist" not in vals and "lcd" not in vals:
            return
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._first_rx is None:
                self._first_rx = now
            self.samples.append((now, vals))

    @property
    def connected(self) -> bool:
        return self._ser is not None

    def status(self, now: float | None = None) -> SerialStatus:
        now = time.monotonic() if now is None else now
        with self._lock:
            recent = [(t, v) for t, v in self.samples if now - t <= self.window_s]
            first = self._first_rx
        dists = [v["dist"] for _, v in recent if "dist" in v]
        lcds = [v["lcd"] for _, v in recent if "lcd" in v]
        receiving = bool(recent) and now - recent[-1][0] <= 2.0
        return SerialStatus(
            connected=self.connected or receiving,
            receiving=receiving,
            last_dist=dists[-1] if dists else None,
            dist_varies=bool(dists) and (max(dists) - min(dists)) > 5,
            sensor_dead=len(dists) >= 5 and all(d <= 0 for d in dists[-5:]),
            lcd_ok=None if not lcds else bool(lcds[-1]),
            seconds_receiving=0.0 if (first is None or not receiving) else now - first,
        )


class FakeSerialMonitor(SerialMonitor):
    """Serial stand-in for simulations/tests: call feed() yourself."""

    def __init__(self):
        super().__init__(port="none")
        self.fake_connected = False

    @property
    def connected(self) -> bool:  # type: ignore[override]
        return self.fake_connected
