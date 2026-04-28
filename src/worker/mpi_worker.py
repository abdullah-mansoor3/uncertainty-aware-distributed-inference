"""MPI-based worker loop that receives tasks from master and executes them.

This file provides a minimal worker process intended to be launched with
`mpirun -n N` alongside the MPI master. Worker ranks (>=1) will block on
receiving messages and execute `generate()` + local attribution, returning
results to the master.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

from mpi4py import MPI

from src.modules.explanation import compute_local_attribution
from src.modules.inference import generate, load_model
from src.utils.config_loader import load_config

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_model_path(project_root: Path, model_path: str) -> str:
    path_obj = Path(model_path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((project_root / path_obj).resolve())


def worker_loop() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # master is rank 0; workers are 1..N-1

    config_path = os.getenv("CLUSTER_CONFIG_PATH", str(PROJECT_ROOT / "configs/cluster_config.yaml"))
    try:
        config = load_config(config_path)
        model_cfg = config["model"]
        worker_cfg = config["nodes"]["worker"]
    except Exception as e:
        LOGGER.exception("Worker failed to load config: %s", e)
        return

    resolved_model_path = _resolve_model_path(PROJECT_ROOT, str(model_cfg["path"]))
    LOGGER.info("Worker %s loading model from %s", rank, resolved_model_path)
    llm = load_model(
        model_path=resolved_model_path,
        n_threads=int(worker_cfg["n_threads"]),
        n_ctx=int(model_cfg["n_ctx"]),
        logprobs=int(model_cfg["logprobs"]),
    )

    LOGGER.info("Worker %s ready", rank)
    comm.send({"type": "ready", "payload": {"worker_rank": rank}}, dest=0, tag=5)

    while True:
        msg = comm.recv(source=0)
        if not isinstance(msg, dict) or "type" not in msg:
            continue
        mtype = msg["type"]
        if mtype == "shutdown":
            LOGGER.info("Worker %s shutting down", rank)
            break
        if mtype == "ping":
            comm.send({"type": "pong"}, dest=0, tag=4)
            continue
        if mtype == "task":
            payload = msg.get("payload", {})
            subtask_text = str(payload.get("text", ""))
            max_tokens = int(payload.get("max_tokens", int(model_cfg["max_tokens"])))
            temperature = float(payload.get("temperature", float(model_cfg.get("temperature", 0.0))))

            t0 = time.perf_counter()
            gen = generate(
                llm=llm,
                prompt=subtask_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=float(model_cfg.get("top_p", 1.0)),
                logprobs=int(model_cfg.get("logprobs", 10)),
                max_inference_ms=int(model_cfg.get("max_inference_ms", 0)),
            )
            t1 = time.perf_counter()
            latency_ms = int((t1 - t0) * 1000)

            output = str(gen.get("text", ""))
            try:
                attribution = compute_local_attribution(subtask=subtask_text, output=output, llm=llm, nlp=None)
            except Exception:
                attribution = []

            result = {
                "id": int(payload.get("id", -1)),
                "output": output,
                "logprobs": [float(v) for v in gen.get("logprobs", [])],
                "tokens": [str(t) for t in gen.get("tokens", [])],
                "attribution": attribution,
                "latency_ms": latency_ms,
                "worker_rank": rank,
            }
            comm.send({"type": "result", "payload": result}, dest=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    worker_loop()
