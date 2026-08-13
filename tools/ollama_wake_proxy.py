#!/usr/bin/env python3
"""Wake-and-proxy for Ollama on a sleeping Mac.

Runs on the client machine (no GPU). Listens on :11434, sends a WoL magic
packet to wake the Mac, waits until Ollama is reachable, then proxies the
request. The Mac must have "Wake for network access" enabled in Battery settings.

Usage:
    python3 ollama_wake_proxy.py --mac-addr AA:BB:CC:DD:EE:FF --mac-ip 192.168.1.42

Then point Ollama clients at http://localhost:11434 as normal.

Config can also be set via environment variables:
    OLLAMA_MAC_ADDR   hardware MAC address of the MacBook NIC
    OLLAMA_MAC_IP     LAN IP of the MacBook
    OLLAMA_MAC_PORT   Ollama port on the Mac (default 11434)
    PROXY_PORT        Port this proxy listens on (default 11434)
"""

import argparse
import os
import socket
import struct
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen, Request
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Config (from CLI args or env)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ollama wake-and-proxy")
    p.add_argument("--mac-addr", default=os.getenv("OLLAMA_MAC_ADDR"),
                   help="Hardware MAC address of the MacBook (AA:BB:CC:DD:EE:FF)")
    p.add_argument("--mac-ip", default=os.getenv("OLLAMA_MAC_IP"),
                   help="LAN IP address of the MacBook")
    p.add_argument("--mac-port", type=int, default=int(os.getenv("OLLAMA_MAC_PORT", "11434")),
                   help="Ollama port on the Mac (default 11434)")
    p.add_argument("--proxy-port", type=int, default=int(os.getenv("PROXY_PORT", "11434")),
                   help="Port this proxy listens on (default 11434)")
    p.add_argument("--wake-timeout", type=int, default=90,
                   help="Seconds to wait for Mac to wake before giving up (default 90)")
    p.add_argument("--wake-interval", type=float, default=3.0,
                   help="Seconds between wake-packet retries (default 3)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Wake-on-LAN
# ---------------------------------------------------------------------------

def send_magic_packet(mac_addr: str, broadcast: str = "255.255.255.255", port: int = 9):
    """Send an Ethernet WoL magic packet (FF×6 + MAC×16)."""
    mac_bytes = bytes.fromhex(mac_addr.replace(":", "").replace("-", ""))
    if len(mac_bytes) != 6:
        raise ValueError(f"Invalid MAC address: {mac_addr}")
    packet = b"\xff" * 6 + mac_bytes * 16
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.sendto(packet, (broadcast, port))


def wait_for_ollama(ip: str, port: int, timeout: int, interval: float,
                    mac_addr: str) -> bool:
    """Send magic packets and poll until Ollama responds or timeout."""
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        # Re-send magic packet every few polls in case the first was missed
        if attempt % 3 == 0:
            try:
                send_magic_packet(mac_addr)
                print(f"[wake] magic packet sent to {mac_addr}")
            except Exception as e:
                print(f"[wake] packet error: {e}")
        attempt += 1
        try:
            with socket.create_connection((ip, port), timeout=2):
                print(f"[wake] Ollama reachable at {ip}:{port}")
                return True
        except OSError:
            remaining = int(deadline - time.monotonic())
            print(f"[wake] waiting for Mac... ({remaining}s remaining)")
            time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Proxy handler
# ---------------------------------------------------------------------------

_config: dict = {}
_wake_lock = threading.Lock()
_mac_awake = False
_last_activity = 0.0
_AWAKE_GRACE = 300  # assume still awake if used within this many seconds


def ensure_awake() -> bool:
    global _mac_awake, _last_activity
    with _wake_lock:
        if _mac_awake and (time.monotonic() - _last_activity) < _AWAKE_GRACE:
            return True
        # Try a quick check first before sending WoL
        try:
            with socket.create_connection(
                (_config["mac_ip"], _config["mac_port"]), timeout=2
            ):
                _mac_awake = True
                _last_activity = time.monotonic()
                return True
        except OSError:
            pass
        print("[wake] Mac appears asleep — sending WoL and waiting...")
        ok = wait_for_ollama(
            _config["mac_ip"],
            _config["mac_port"],
            _config["wake_timeout"],
            _config["wake_interval"],
            _config["mac_addr"],
        )
        if ok:
            _mac_awake = True
            _last_activity = time.monotonic()
        else:
            _mac_awake = False
        return ok


class OllamaProxy(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[proxy] {self.address_string()} - {fmt % args}")

    def _proxy(self):
        global _last_activity
        if not ensure_awake():
            self.send_error(503, "Mac did not wake in time")
            return

        target = (
            f"http://{_config['mac_ip']}:{_config['mac_port']}{self.path}"
        )
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else None

        req = Request(target, data=body, method=self.command)
        for key, val in self.headers.items():
            if key.lower() not in ("host", "content-length"):
                req.add_header(key, val)
        if body:
            req.add_header("Content-Length", str(len(body)))

        try:
            with urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                for key, val in resp.headers.items():
                    if key.lower() not in ("transfer-encoding",):
                        self.send_header(key, val)
                self.end_headers()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                _last_activity = time.monotonic()
        except URLError as e:
            self.send_error(502, f"Upstream error: {e}")

    def do_GET(self):    self._proxy()
    def do_POST(self):   self._proxy()
    def do_DELETE(self): self._proxy()
    def do_HEAD(self):   self._proxy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if not args.mac_addr:
        raise SystemExit("--mac-addr is required (or set OLLAMA_MAC_ADDR)")
    if not args.mac_ip:
        raise SystemExit("--mac-ip is required (or set OLLAMA_MAC_IP)")

    _config.update({
        "mac_addr": args.mac_addr,
        "mac_ip": args.mac_ip,
        "mac_port": args.mac_port,
        "wake_timeout": args.wake_timeout,
        "wake_interval": args.wake_interval,
    })

    print(f"Ollama wake-proxy listening on :{args.proxy_port}")
    print(f"  Target: {args.mac_ip}:{args.mac_port}  (MAC {args.mac_addr})")
    print(f"  Wake timeout: {args.wake_timeout}s")

    server = HTTPServer(("0.0.0.0", args.proxy_port), OllamaProxy)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
