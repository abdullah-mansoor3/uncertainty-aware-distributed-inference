"""Networking utilities for node health checks and subtask dispatch."""

from __future__ import annotations

import asyncio
import platform
import subprocess
from typing import Any, Dict

import httpx


def ping_node(ip: str) -> float:
    """Ping a node and return RTT in milliseconds, or -1 if unavailable."""
    if not isinstance(ip, str) or not ip.strip():
        raise ValueError("ip must be a non-empty string")

    count_flag = "-n" if platform.system().lower() == "windows" else "-c"
    command = ["ping", count_flag, "1", ip]

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
    except Exception:
        return -1.0

    output = (result.stdout or "") + (result.stderr or "")
    marker = "time="
    idx = output.find(marker)
    if idx == -1:
        return -1.0

    rest = output[idx + len(marker) :]
    number = ""
    for ch in rest:
        if ch.isdigit() or ch == ".":
            number += ch
        else:
            break

    try:
        return float(number)
    except Exception:
        return -1.0


async def send_subtask(ip: str, port: int, subtask: Dict[str, Any], timeout_s: float = 30.0) -> Dict[str, Any]:
    """Send a subtask to worker /infer endpoint."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    url = f"http://{ip}:{port}/infer"
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(url, json=subtask)
        response.raise_for_status()
        return response.json()


def check_network_feasibility(ip: str, threshold_ms: float = 100.0) -> bool:
    """Check if network RTT is below threshold for parallel execution."""
    rtt = ping_node(ip)
    return rtt >= 0.0 and rtt <= threshold_ms
