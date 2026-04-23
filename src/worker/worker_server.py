"""FastAPI worker for remote low-uncertainty subtask execution.

Implements the node_b inference service used by naive and uncertainty-aware
pipelines. Inference is served via llama-cpp-python; local token attribution
uses the SyntaxShap-style approximation from the explanation module.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.modules.explanation import compute_local_attribution
from src.modules.inference import generate, load_model
from src.utils.config_loader import load_config

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class InferRequest(BaseModel):
    """Payload for /infer worker requests.

    Args:
        subtask_text: Subtask prompt text.
        max_tokens: Maximum output tokens.
        temperature: Generation temperature.
    """

    subtask_text: str = Field(..., min_length=1)
    max_tokens: int = Field(256, gt=0)
    temperature: float = Field(0.6, ge=0.0)


class InferResponse(BaseModel):
    """Response model for /infer endpoint."""

    output: str
    logprobs: List[float]
    tokens: List[str]
    attribution: List[Dict[str, float]]
    latency_ms: int


def _resolve_model_path(project_root: Path, model_path: str) -> str:
    """Resolve model path from config against the project root."""
    path_obj = Path(model_path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((project_root / path_obj).resolve())


def _load_spacy_pipeline() -> Any:
    """Load spaCy parser; fallback to a lightweight blank English pipeline."""
    try:
        return spacy.load("en_core_web_sm")
    except Exception as error:
        LOGGER.warning("Failed to load en_core_web_sm (%s). Falling back to spacy.blank('en').", error)
        return spacy.blank("en")


@asynccontextmanager
async def lifespan(app_obj: FastAPI):
    """Startup/shutdown lifecycle for model and NLP pipeline initialization."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    app_obj.state.model_loaded = False
    app_obj.state.startup_error = None
    app_obj.state.llm = None
    app_obj.state.nlp = None
    app_obj.state.model_settings = {}

    config_path = os.getenv("CLUSTER_CONFIG_PATH", str(PROJECT_ROOT / "configs/cluster_config.yaml"))
    try:
        config = load_config(config_path)
        model_config = config["model"]
        worker_config = config["nodes"]["worker"]

        resolved_model_path = _resolve_model_path(PROJECT_ROOT, str(model_config["path"]))
        app_obj.state.llm = load_model(
            model_path=resolved_model_path,
            n_threads=int(worker_config["n_threads"]),
            n_ctx=int(model_config["n_ctx"]),
            logprobs=int(model_config["logprobs"]),
        )
        app_obj.state.nlp = _load_spacy_pipeline()
        app_obj.state.model_settings = {
            "top_p": float(model_config["top_p"]),
            "logprobs": int(model_config["logprobs"]),
            "max_inference_ms": int(model_config["max_inference_ms"]),
        }
        app_obj.state.model_loaded = True
        LOGGER.info("Worker model initialized from %s", resolved_model_path)
    except Exception as error:  # pragma: no cover - startup failure path
        app_obj.state.startup_error = str(error)
        LOGGER.exception("Worker startup failed: %s", error)

    yield


app = FastAPI(title="Worker Inference Server", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health endpoint including model-load status for scheduler checks."""
    return {
        "status": "ok",
        "node": "i5",
        "model_loaded": bool(getattr(app.state, "model_loaded", False)),
        "startup_error": getattr(app.state, "startup_error", None),
    }


@app.post("/infer", response_model=InferResponse)
def infer(payload: InferRequest) -> InferResponse:
    """Run worker inference and local attribution for a single subtask request."""
    if not bool(getattr(app.state, "model_loaded", False)):
        raise HTTPException(status_code=503, detail="Worker model is not loaded")

    llm = app.state.llm
    nlp = app.state.nlp
    settings = dict(getattr(app.state, "model_settings", {}))

    result = generate(
        llm=llm,
        prompt=payload.subtask_text,
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_p=float(settings.get("top_p", 0.9)),
        logprobs=int(settings.get("logprobs", 10)),
        max_inference_ms=int(settings.get("max_inference_ms", 120000)),
    )

    output_text = str(result.get("text", ""))
    try:
        attribution = compute_local_attribution(subtask=payload.subtask_text, output=output_text, llm=llm, nlp=nlp)
    except Exception as error:  # pragma: no cover - non-critical attribution path
        LOGGER.warning("Attribution failed on worker: %s", error)
        attribution = []

    return InferResponse(
        output=output_text,
        logprobs=[float(value) for value in result.get("logprobs", [])],
        tokens=[str(token) for token in result.get("tokens", [])],
        attribution=[
            {"token": str(item.get("token", "")), "attribution": float(item.get("attribution", 0.0))}
            for item in attribution
            if isinstance(item, dict)
        ],
        latency_ms=int(result.get("latency_ms", 0)),
    )
