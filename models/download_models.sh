#!/usr/bin/env bash
set -euo pipefail

# Verified source:
#   Repo: bartowski/Llama-3.2-3B-Instruct-GGUF
#   File: Llama-3.2-3B-Instruct-Q4_K_M.gguf
# Saved locally to the canonical project path expected by cluster_config.yaml.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${SCRIPT_DIR}/llama-3.2-3b-q4_k_m.gguf"
MODEL_REPO="bartowski/Llama-3.2-3B-Instruct-GGUF"
MODEL_FILE="Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"

echo "Target: ${MODEL_PATH}"
echo "Source: ${MODEL_URL}"

mkdir -p "$(dirname "${MODEL_PATH}")"

if [[ -s "${MODEL_PATH}" ]]; then
	echo "Model already exists and is non-empty. Skipping download."
	exit 0
fi

AUTH_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
	AUTH_ARGS=(-H "Authorization: Bearer ${HF_TOKEN}")
	echo "Using HF_TOKEN for authenticated download."
fi

if command -v curl >/dev/null 2>&1; then
	curl "${AUTH_ARGS[@]}" \
		--fail \
		--location \
		--retry 5 \
		--retry-all-errors \
		--retry-delay 2 \
		--continue-at - \
		--output "${MODEL_PATH}" \
		"${MODEL_URL}"
elif command -v wget >/dev/null 2>&1; then
	WGET_AUTH=()
	if [[ -n "${HF_TOKEN:-}" ]]; then
		WGET_AUTH=(--header="Authorization: Bearer ${HF_TOKEN}")
	fi
	wget "${WGET_AUTH[@]}" \
		--tries=5 \
		--continue \
		--output-document="${MODEL_PATH}" \
		"${MODEL_URL}"
else
	echo "Error: neither curl nor wget is installed."
	exit 1
fi

if [[ ! -s "${MODEL_PATH}" ]]; then
	echo "Error: download completed but file is missing or empty: ${MODEL_PATH}"
	exit 1
fi

echo "Model downloaded successfully to: ${MODEL_PATH}"
