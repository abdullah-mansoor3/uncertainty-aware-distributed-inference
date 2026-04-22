<<<<<<< HEAD
"""Networking utilities for node health checks and subtask dispatch."""
=======
"""Networking utilities for cluster communication.

Provides RTT checks and worker request helpers over HTTP for distributed
execution. Serial baseline uses only health/feasibility checks.
"""
>>>>>>> 2c641dd (feat: Full project scaffold)

from __future__ import annotations

import asyncio
<<<<<<< HEAD
import platform
import subprocess
=======
import time
>>>>>>> 2c641dd (feat: Full project scaffold)
from typing import Any, Dict

import httpx


<<<<<<< HEAD
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
=======
def ping_node(ip: str, timeout_s: float = 2.0) -> float:
    """Ping worker health endpoint and return RTT in milliseconds.

    Args:
        ip: Node IP or host.
        timeout_s: Request timeout in seconds.

    Returns:
        RTT in ms, or -1.0 when unreachable.
    """
    if not ip:
        raise ValueError("ip must be a non-empty string")

    started_at = time.perf_counter()
    url = f"http://{ip}:8001/health"
    try:
        response = httpx.get(url, timeout=timeout_s)
        if response.status_code != 200:
            return -1.0
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return float(elapsed_ms)
>>>>>>> 2c641dd (feat: Full project scaffold)
    except Exception:
        return -1.0


<<<<<<< HEAD
async def send_subtask(ip: str, port: int, subtask: Dict[str, Any], timeout_s: float = 30.0) -> Dict[str, Any]:
    """Send a subtask to worker /infer endpoint."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
=======
async def send_subtask(
    ip: str,
    port: int,
    subtask: Dict[str, Any],
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Send subtask to worker infer endpoint.

    Args:
        ip: Worker IP.
        port: Worker port.
        subtask: JSON payload with subtask request params.
        timeout_s: HTTP timeout in seconds.

    Returns:
        Parsed JSON response dict.
    """
    if not ip:
        raise ValueError("ip must be non-empty")
    if port <= 0:
        raise ValueError("port must be > 0")
    if not isinstance(subtask, dict):
        raise ValueError("subtask must be a dictionary")
>>>>>>> 2c641dd (feat: Full project scaffold)

    url = f"http://{ip}:{port}/infer"
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(url, json=subtask)
        response.raise_for_status()
        return response.json()


def check_network_feasibility(ip: str, threshold_ms: float = 100.0) -> bool:
<<<<<<< HEAD
    """Check if network RTT is below threshold for parallel execution."""
    rtt = ping_node(ip)
    return rtt >= 0.0 and rtt <= threshold_ms
=======
    """Check if RTT is within acceptable threshold.

    Args:
        ip: Node IP.
        threshold_ms: Maximum acceptable RTT in milliseconds.

    Returns:
        True when reachable and under threshold.
    """
    if threshold_ms <= 0:
        raise ValueError("threshold_ms must be > 0")

    rtt_ms = ping_node(ip)
    return rtt_ms >= 0.0 and rtt_ms <= threshold_ms


def send_subtask_sync(ip: str, port: int, subtask: Dict[str, Any], timeout_s: float = 30.0) -> Dict[str, Any]:
    """Synchronous wrapper around async worker subtask sender.

    Args:
        ip: Worker IP.
        port: Worker port.
        subtask: Request payload.
        timeout_s: Timeout in seconds.

    Returns:
        Worker JSON result.
    """
    return asyncio.run(send_subtask(ip=ip, port=port, subtask=subtask, timeout_s=timeout_s))
>>>>>>> 2c641dd (feat: Full project scaffold)
