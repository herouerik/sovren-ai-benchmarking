"""Mid-flight thrash detection for streaming completions.

The point of this guard is to *save wall-clock time*, so it must abort a call
while it is still running. Classifying a sample as bad after it returns is too
late: a thrashing generation runs effectively forever.

Two regimes, because they need different signals:

  prefill (no tokens yet)
      A long TTFT is legitimate — cold model load and extended thinking both
      look like silence. Nothing is killed on slowness alone here; only a
      sustained OS swap rate trips it, plus a very generous absolute ceiling
      as a backstop.

  decode (tokens flowing)
      Thrash is unmistakable: inter-token gaps blow out. Two triggers — a hard
      stall (no token at all for N seconds) and a rate collapse relative to the
      rate this same call established over its first tokens. Being relative,
      a consistently slow model never trips it while a swapping one always does.

The OS sensor (harness/pressure.py) never triggers a decode abort on its own;
it grades one. `hard` means the pressure was corroborated — the model does not
fit, so skip its remaining benchmarks. `soft` means timing alone tripped, which
costs one sample and lets the run continue.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from harness.pressure import PressureSensor, get_sensor

DEFAULTS = {
    "enabled": True,
    "poll_interval": 1.0,
    # prefill: generous — cold load and thinking are legitimate
    "ttft_ceiling_seconds": 300.0,
    # decode: absolute silence that no healthy generation exhibits
    "token_stall_seconds": 20.0,
    # decode: collapse relative to this call's own established rate.
    # Rates are measured over wall-clock windows, never per-token gaps: tokens
    # arrive in bursts (several per network read, and many per step under
    # speculative decoding), so consecutive gaps can be microseconds and imply
    # absurd rates like 20000 tok/s. Averaging over a window is burst-immune.
    "decode_degradation_factor": 10.0,
    "baseline_window_seconds": 5.0,   # observe this long to establish the rate
    "baseline_tokens": 16,            # ...and require at least this many tokens
    "window_seconds": 5.0,            # trailing window compared against it
    # OS corroboration: sustained swap-out rate that means real thrash
    "swap_bytes_per_sec": 32 * 1024 * 1024,
    "swap_sustain_seconds": 5.0,
    # Throughput floor — a separate question from thrash: a model can decode
    # perfectly steadily and still be too slow to be worth benchmarking on this
    # machine (a 30B at 0.2 tok/s on a CPU laptop is healthy, just useless).
    # 0 disables. Set per-machine, not per-model: it describes the hardware.
    "min_decode_tps": 0.0,
    "min_decode_tps_after_tokens": 32,
}


def _median(values: list[float]) -> float:
    clean = sorted(values)
    n = len(clean)
    if n == 0:
        return 0.0
    return clean[n // 2] if n % 2 else (clean[n // 2 - 1] + clean[n // 2]) / 2


class StreamWatchdog:
    """Watches one streaming generation and trips when it stalls or thrashes."""

    def __init__(self, cfg: dict | None = None, sensor: PressureSensor | None = None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.sensor = sensor if sensor is not None else get_sensor()

        self.reason: str | None = None
        self.hard = False
        self.kind: str | None = None

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_trip: Callable[[], None] | None = None

        self._t_start = 0.0
        self._t_first: float | None = None
        self._t_last = 0.0
        self._token_times: list[float] = []
        self._baseline_rate: float | None = None

        self._prev_swap: tuple[int, int] | None = None
        self._swap_since: float | None = None

    # ── streaming-thread API ────────────────────────────────────────────────

    def begin(self) -> None:
        """Mark the start of the request (before the HTTP call is made)."""
        self._t_start = time.perf_counter()
        self._t_last = self._t_start
        self._prev_swap = self.sensor.swap_bytes()

    def arm(self, on_trip: Callable[[], None]) -> None:
        """Start watching. `on_trip` must close the stream to unblock the reader."""
        self._on_trip = on_trip
        if not self.cfg.get("enabled", True):
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def on_token(self) -> None:
        """Record one token's arrival. Called for answer *and* thinking tokens."""
        now = time.perf_counter()
        with self._lock:
            if self._t_first is None:
                self._t_first = now
            self._token_times.append(now)
            self._t_last = now

    def stop(self) -> None:
        self._stop.set()

    @property
    def tripped(self) -> bool:
        return self.reason is not None

    # ── watchdog thread ─────────────────────────────────────────────────────

    def _loop(self) -> None:
        cfg = self.cfg
        while not self._stop.wait(cfg["poll_interval"]):
            now = time.perf_counter()
            swapping = self._swap_active(now)
            with self._lock:
                t_first, t_last = self._t_first, self._t_last
                times = list(self._token_times)

            if t_first is None:
                waited = now - self._t_start
                if swapping:
                    self._trip(f"sustained swap during prefill after {waited:.1f}s", hard=True)
                    return
                if waited > cfg["ttft_ceiling_seconds"]:
                    self._trip(
                        f"no first token after {waited:.0f}s "
                        f"(ttft_ceiling_seconds={cfg['ttft_ceiling_seconds']:.0f})",
                        hard=False,
                    )
                    return
                continue

            since_last = now - t_last
            if since_last > cfg["token_stall_seconds"]:
                self._trip(
                    f"decode stalled {since_last:.1f}s without a token "
                    f"(token_stall_seconds={cfg['token_stall_seconds']:.0f})",
                    hard=swapping,
                )
                return

            # Sustained rate over the whole decode so far, for the "too slow
            # to bother" floor. Burst arrival averages out over this span.
            span = t_last - t_first
            sustained = (len(times) - 1) / span if span > 0 and len(times) > 1 else 0.0

            floor = cfg.get("min_decode_tps", 0.0)
            if (floor > 0 and len(times) >= cfg["min_decode_tps_after_tokens"]
                    and span >= cfg["baseline_window_seconds"] and 0 < sustained < floor):
                self._trip(
                    f"sustained {sustained:.2f} tok/s is below "
                    f"min_decode_tps={floor} — too slow to benchmark here",
                    hard=True, kind="too_slow",
                )
                return

            # Establish the reference rate once, over a wall-clock window.
            if self._baseline_rate is None:
                if span >= cfg["baseline_window_seconds"] and len(times) >= cfg["baseline_tokens"]:
                    self._baseline_rate = sustained
                continue

            # Compare a trailing window against it.
            window = cfg["window_seconds"]
            if now - t_first < cfg["baseline_window_seconds"] + window:
                continue
            recent_n = sum(1 for t in times if now - t <= window)
            recent_rate = recent_n / window
            if (self._baseline_rate > 0
                    and recent_rate * cfg["decode_degradation_factor"] < self._baseline_rate):
                self._trip(
                    f"decode collapsed {self._baseline_rate / max(recent_rate, 1e-9):.0f}x "
                    f"({self._baseline_rate:.0f} -> {recent_rate:.1f} tok/s)",
                    hard=swapping,
                )
                return

    def _swap_active(self, now: float) -> bool:
        """True once the swap-out rate has stayed above threshold long enough."""
        current = self.sensor.swap_bytes()
        if current is None or self._prev_swap is None:
            self._prev_swap = current
            return False
        delta_out = current[1] - self._prev_swap[1]
        self._prev_swap = current
        rate = delta_out / max(self.cfg["poll_interval"], 1e-6)
        if rate >= self.cfg["swap_bytes_per_sec"]:
            if self._swap_since is None:
                self._swap_since = now
            return (now - self._swap_since) >= self.cfg["swap_sustain_seconds"]
        self._swap_since = None
        return False

    def _trip(self, reason: str, hard: bool, kind: str = "swap") -> None:
        # The prefix is what the dashboard greps for, so "too slow to be worth
        # running" stays visibly distinct from "the machine was thrashing"
        # rather than being reported as a memory problem it isn't.
        self.kind = kind
        self.reason = f"{'swap_abort' if kind == 'swap' else kind}: {reason}"
        self.hard = hard
        if self._on_trip:
            try:
                self._on_trip()
            except Exception:
                pass  # closing a stream mid-flight is best-effort
