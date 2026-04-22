<<<<<<< HEAD
"""Inference wrapper module for llama-cpp-python.

Provides a unified interface for local generation on both cluster nodes.
=======
"""Inference wrapper over llama-cpp-python for CPU-only GGUF execution.

This module centralizes model loading and generation behavior for both master and
worker roles while preserving token-level log-probability outputs for PRO scoring.
>>>>>>> 2c641dd (feat: Full project scaffold)
"""

from __future__ import annotations

<<<<<<< HEAD
import time
from typing import Any, Dict

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - allows skeleton import before dependency install.
    Llama = Any


def load_model(model_path: str, n_threads: int, n_ctx: int = 2048) -> Llama:
    """Load a GGUF model using llama-cpp-python.

    Args:
        model_path: Path to model file.
        n_threads: CPU thread count.
        n_ctx: Context length.

    Returns:
        Initialized llama model instance.
    """
    if not model_path:
        raise ValueError("model_path must be provided")
    if n_threads <= 0:
        raise ValueError("n_threads must be positive")

    return Llama(model_path=model_path, n_threads=n_threads, n_ctx=n_ctx, logprobs=10)
=======
import logging
import time
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - depends on local install
    Llama = None  # type: ignore[assignment]


def load_model(model_path: str, n_threads: int, n_ctx: int = 2048, logprobs: int = 10) -> Llama:
    """Load GGUF model with node-specific CPU configuration.

    Args:
        model_path: Filesystem path to GGUF model.
        n_threads: Number of CPU threads for inference.
        n_ctx: Context length.
        logprobs: Top-K logprobs to return in completion.

    Returns:
        Initialized Llama model instance.
    """
    if not model_path:
        raise ValueError("model_path must be a non-empty string")
    if n_threads <= 0:
        raise ValueError("n_threads must be > 0")
    if Llama is None:
        raise ImportError(
            "llama-cpp-python is not installed. Install dependencies with: pip install -r requirements.txt"
        )

    return Llama(model_path=model_path, n_threads=n_threads, n_ctx=n_ctx, logits_all=True, verbose=False)
>>>>>>> 2c641dd (feat: Full project scaffold)


def generate(
    llm: Llama,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.9,
<<<<<<< HEAD
) -> Dict[str, Any]:
    """Generate text and return token log-probability metadata.

    Args:
        llm: Loaded llama model.
        prompt: Input prompt.
        max_tokens: Maximum generation tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.

    Returns:
        A dictionary containing text, logprobs, tokens, and latency_ms.
=======
    logprobs: int = 10,
    max_inference_ms: int = 120000,
) -> Dict[str, Any]:
    """Run text generation and always return text, tokens, logprobs, and latency.

    Args:
        llm: Loaded model object.
        prompt: Prompt text.
        max_tokens: Max decode tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        logprobs: Number of top logprobs requested.
        max_inference_ms: Timeout threshold for inference duration.

    Returns:
        Dict containing generated text, logprobs, tokens, and latency.
>>>>>>> 2c641dd (feat: Full project scaffold)
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

<<<<<<< HEAD
    start = time.perf_counter()
    completion = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=10,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    choice = completion["choices"][0]
    text = choice.get("text", "")
    logprobs_block = choice.get("logprobs") or {}
    token_logprobs = logprobs_block.get("token_logprobs") or []
    tokens = logprobs_block.get("tokens") or []

    return {
        "text": text,
        "logprobs": token_logprobs,
        "tokens": tokens,
        "latency_ms": latency_ms,
    }
=======
    started_at = time.perf_counter()
    try:
        completion = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)

        choice = completion["choices"][0]
        completion_logprobs = choice.get("logprobs", {})
        token_logprobs: List[float] = completion_logprobs.get("token_logprobs") or []
        top_logprobs: List[Dict[str, float]] = completion_logprobs.get("top_logprobs") or []
        tokens: List[str] = completion_logprobs.get("tokens") or []

        pro_logprobs: List[float] = []
        if top_logprobs and isinstance(top_logprobs[0], dict):
            sorted_items = sorted(top_logprobs[0].items(), key=lambda item: item[1], reverse=True)
            pro_logprobs = [float(logp) for _, logp in sorted_items[:logprobs]]
        elif token_logprobs:
            pro_logprobs = [float(value) for value in token_logprobs[:logprobs] if value is not None]

        if elapsed_ms > max_inference_ms:
            LOGGER.warning("Inference timeout exceeded: %sms > %sms", elapsed_ms, max_inference_ms)

        return {
            "text": choice.get("text", "").strip(),
            "logprobs": [float(value) for value in token_logprobs if value is not None],
            "pro_logprobs": pro_logprobs,
            "top_logprobs": top_logprobs,
            "tokens": [str(token) for token in tokens],
            "latency_ms": elapsed_ms,
            "timed_out": elapsed_ms > max_inference_ms,
        }
    except Exception as error:
        LOGGER.exception("Inference call failed: %s", error)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "text": "",
            "logprobs": [],
            "pro_logprobs": [],
            "top_logprobs": [],
            "tokens": [],
            "latency_ms": elapsed_ms,
            "timed_out": False,
            "error": str(error),
        }
>>>>>>> 2c641dd (feat: Full project scaffold)
