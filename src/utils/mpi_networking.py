"""Simple MPI networking helpers for mpi4py-based runners.

This module defines a lightweight message protocol used by the MPI runners:
- Master (rank 0) sends messages of type 'task' with a payload dict containing
  subtask fields. Workers (ranks >=1) receive, execute, and reply with a
  'result' message containing execution output and telemetry.

The implementation is intentionally minimal and uses `comm.send`/`comm.recv`
to remain compatible with OpenMPI launch patterns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def worker_ranks() -> List[int]:
    """Return list of worker ranks (1..size-1)."""
    return list(range(1, size)) if size > 1 else []


def send_task(to_rank: int, task: Dict[str, Any]) -> None:
    """Send a task dict to a worker rank."""
    comm.send({'type': 'task', 'payload': task}, dest=to_rank, tag=0)


def recv_message(source: Optional[int] = MPI.ANY_SOURCE, tag: int = MPI.ANY_TAG) -> Dict[str, Any]:
    """Receive a message from any source (blocking)."""
    return comm.recv(source=source, tag=tag)


def send_result(to_rank: int, result: Dict[str, Any]) -> None:
    """Send a result dict back to the master."""
    comm.send({'type': 'result', 'payload': result}, dest=to_rank, tag=1)


def broadcast_shutdown() -> None:
    """Tell all workers to shutdown (master only)."""
    for r in worker_ranks():
        comm.send({'type': 'shutdown'}, dest=r, tag=2)


def probe_workers() -> Dict[int, float]:
    """Quick ping: ask each worker for a heartbeat and return RTT estimate per worker in ms.

    The function sends a ping and waits for a pong; measured on master only.
    """
    if rank != 0:
        return {}
    rtts: Dict[int, float] = {}
    import time

    for r in worker_ranks():
        t0 = time.perf_counter()
        comm.send({'type': 'ping'}, dest=r, tag=3)
        # expect a simple {'type':'pong'}
        msg = comm.recv(source=r, tag=4)
        t1 = time.perf_counter()
        rtts[r] = (t1 - t0) * 1000.0
    return rtts


def wait_for_workers(expected_workers: List[int], timeout_s: float = 120.0) -> List[int]:
    """Wait for workers to send a readiness message.

    Args:
        expected_workers: Worker ranks to wait for.
        timeout_s: Max seconds to wait before returning.

    Returns:
        List of worker ranks that signaled readiness.
    """
    if rank != 0:
        return []

    import time

    pending = set(int(r) for r in expected_workers)
    ready: List[int] = []
    started = time.perf_counter()

    while pending and (time.perf_counter() - started) < timeout_s:
        if comm.Iprobe(source=MPI.ANY_SOURCE, tag=5):
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=5)
            if isinstance(msg, dict) and msg.get("type") == "ready":
                payload = msg.get("payload", {})
                rank_id = int(payload.get("worker_rank", -1))
                if rank_id in pending:
                    pending.remove(rank_id)
                    ready.append(rank_id)
        else:
            time.sleep(0.05)

    return sorted(ready)
