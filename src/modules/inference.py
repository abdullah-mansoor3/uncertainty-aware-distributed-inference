"""Inference wrapper module for llama-cpp-python.

Provides a unified interface for local generation on both cluster nodes.
"""

from __future__ import annotations

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


def generate(
    llm: Llama,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.9,
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
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

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
