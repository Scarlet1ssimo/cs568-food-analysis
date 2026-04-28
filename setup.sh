# NOTE: uv is required as a Python package manager, checkout https://docs.astral.sh/uv/getting-started/installation/
uv sync
source .venv/bin/activate # This should give you `kaggle` CLI.

mkdir -p data/raw data/processed
kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions -p data/raw --unzip

# NOTE: You need to create a .env file with Gemini API key. Checkout https://aistudio.google.com/api-keys
# Gemma 3 27B is available and free at a rate limit of 30 RPM (round per minute), 15K TPM (tokens per minute), and 14.4k RPD (requests per day).
cp .env.example .env

uv run main.py