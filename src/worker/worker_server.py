"""FastAPI worker server for remote subtask inference on node_b."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Worker Inference Server", version="0.1.0")


class InferRequest(BaseModel):
    """Payload received by POST /infer."""

    subtask_text: str
    max_tokens: int = 256
    temperature: float = 0.6


class InferResponse(BaseModel):
    """Response returned by POST /infer."""

    output: str
    logprobs: List[float]
    tokens: List[str]
    attribution: List[Dict[str, Any]]
    latency_ms: int


@app.get("/health")
def health() -> Dict[str, Any]:
    """Simple liveness endpoint."""
    return {"status": "ok", "node": "i5", "model_loaded": False}


@app.post("/infer", response_model=InferResponse)
def infer(payload: InferRequest) -> InferResponse:
    """Inference endpoint scaffold.

    The full implementation should load model once at startup and run generation
    plus local attribution per request.
    """
    text = payload.subtask_text.strip()
    return InferResponse(
        output=text,
        logprobs=[],
        tokens=text.split(),
        attribution=[],
        latency_ms=0,
    )
