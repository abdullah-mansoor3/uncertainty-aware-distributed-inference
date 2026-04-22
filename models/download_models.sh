#!/usr/bin/env bash
set -euo pipefail

<<<<<<< HEAD
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
=======
# Downloads Llama-3.2-3B Instruct Q4_K_M GGUF into ./models/.
# Requires a Hugging Face token with accepted Meta license access.

MODEL_REPO="bartowski/Llama-3.2-3B-Instruct-GGUF"
MODEL_FILE="Llama-3.2-3B-Instruct-Q4_K_M.gguf"
TARGET_FILE="llama-3.2-3b-instruct-q4_k_m.gguf"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_CLI=""
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif [[ -x "$HOME/.local/bin/hf" ]]; then
  HF_CLI="$HOME/.local/bin/hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
else
  echo "Hugging Face CLI not found. Install with: pip install huggingface_hub"
  echo "Then ensure ~/.local/bin is on PATH."
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export your token first:"
  echo "  export HF_TOKEN=hf_xxx"
  exit 1
fi

echo "Downloading ${MODEL_FILE} from ${MODEL_REPO} ..."
if [[ "${HF_CLI}" == *"huggingface-cli" ]]; then
  "${HF_CLI}" download "${MODEL_REPO}" "${MODEL_FILE}" \
    --local-dir "${OUT_DIR}" \
    --token "${HF_TOKEN}"
else
  "${HF_CLI}" download "${MODEL_REPO}" "${MODEL_FILE}" \
    --local-dir "${OUT_DIR}" \
    --token "${HF_TOKEN}"
fi

if [[ -f "${OUT_DIR}/${MODEL_FILE}" ]]; then
  mv "${OUT_DIR}/${MODEL_FILE}" "${OUT_DIR}/${TARGET_FILE}"
fi

echo "Done. Model at: ${OUT_DIR}/${TARGET_FILE}"
>>>>>>> 2c641dd (feat: Full project scaffold)
