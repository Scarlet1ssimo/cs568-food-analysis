#!/usr/bin/env sh

# NOTE: uv is required as a Python package manager.
# Install docs: https://docs.astral.sh/uv/getting-started/installation/

set -eu

if ! command -v uv >/dev/null 2>&1; then
	echo "Error: uv is not installed or not on PATH."
	exit 1
fi

uv sync

mkdir -p data/raw data/processed

# Use uv run to execute kaggle from the project environment in a shell-agnostic way.
uv run kaggle datasets download \
	-d shuyangli94/food-com-recipes-and-user-interactions \
	-p data/raw \
	--unzip

# NOTE: Create a .env file with your LLM provider API key(s).
# Gemini key docs: https://aistudio.google.com/api-keys
# OpenAI key docs: https://platform.openai.com/api-keys
cp .env.sample .env

# Example runs:
# uv run cs568-prepare --llm-provider google --model-name gemini-2.5-flash
# uv run cs568-prepare --llm-provider openai --model-name gpt-4o-mini
# uv run cs568-analyze