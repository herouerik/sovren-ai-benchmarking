"""Portable memory-pressure sensing for the streaming swap guard.

Timing is the *primary* thrash signal (see harness/guard.py) because it is the
only one that works everywhere. The failure mode differs by platform:

    macOS / Apple Silicon  unified memory oversubscribed -> OS swap thrash
    Linux + NVIDIA         VRAM exhausted -> layers offload to CPU (no swap!)
    Linux, CPU-only        host RAM oversubscribed -> OS swap thrash

Only the first and third show up in swap counters, so a swap-counter guard
would sit there reporting "healthy" on a GPU box while decode crawls. These
sensors therefore only *corroborate* a timing trip: they decide whether the
trip means "this model does not fit at all" (skip the model) or "one bad
sample" (drop the sample and continue).

Adding a platform is one subclass; everything degrades to timing-only when no
sensor is available.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path


class PressureSensor:
    """No-op sensor. Used when the platform is unknown or counters are absent."""

    name = "none"
    available = False

    def swap_bytes(self) -> tuple[int, int] | None:
        """Cumulative (swapped_in, swapped_out) bytes, or None if unavailable."""
        return None


class DarwinSensor(PressureSensor):
    """macOS: `vm_stat` exposes cumulative Swapins/Swapouts in pages (~4ms/read)."""

    name = "darwin"
    _PAGE_RE = re.compile(r"page size of (\d+) bytes")

    def __init__(self) -> None:
        self.page_size = 4096
        out = self._read()
        if out:
            m = self._PAGE_RE.search(out)
            if m:
                self.page_size = int(m.group(1))
            self.available = "Swapouts" in out

    @staticmethod
    def _read() -> str | None:
        try:
            return subprocess.run(["vm_stat"], capture_output=True, text=True,
                                  timeout=5).stdout
        except Exception:
            return None

    def swap_bytes(self) -> tuple[int, int] | None:
        out = self._read()
        if not out:
            return None
        vals: dict[str, int] = {}
        for line in out.splitlines():
            if ":" not in line:
                continue
            key, _, raw = line.partition(":")
            raw = raw.strip().rstrip(".")
            if raw.isdigit():
                vals[key.strip()] = int(raw)
        si, so = vals.get("Swapins"), vals.get("Swapouts")
        if si is None or so is None:
            return None
        return si * self.page_size, so * self.page_size


class LinuxSensor(PressureSensor):
    """Linux: /proc/vmstat exposes cumulative pswpin/pswpout in pages."""

    name = "linux"
    PATH = Path("/proc/vmstat")

    def __init__(self) -> None:
        try:
            self.page_size = os.sysconf("SC_PAGE_SIZE")
        except Exception:
            self.page_size = 4096
        self.available = self.PATH.exists() and self.swap_bytes() is not None

    def swap_bytes(self) -> tuple[int, int] | None:
        try:
            text = self.PATH.read_text()
        except Exception:
            return None
        vals: dict[str, int] = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in ("pswpin", "pswpout"):
                try:
                    vals[parts[0]] = int(parts[1])
                except ValueError:
                    pass
        if "pswpin" not in vals or "pswpout" not in vals:
            return None
        return vals["pswpin"] * self.page_size, vals["pswpout"] * self.page_size


def get_sensor() -> PressureSensor:
    """Return the best available sensor for this platform (never raises)."""
    system = platform.system()
    try:
        if system == "Darwin":
            sensor: PressureSensor = DarwinSensor()
        elif system == "Linux":
            sensor = LinuxSensor()
        else:
            return PressureSensor()
    except Exception:
        return PressureSensor()
    return sensor if sensor.available else PressureSensor()
