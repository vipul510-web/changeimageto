#!/bin/bash
# Script to start backend with Dodo Payments and Replicate environment variables

# Set environment variables (replace with your actual keys)
# These should be set in your environment or passed when running the script
export DODO_PAYMENTS_API_KEY="${DODO_PAYMENTS_API_KEY:-your-dodo-payments-api-key-here}"
export DODO_PAYMENTS_PRODUCT_ID="${DODO_PAYMENTS_PRODUCT_ID:-your-dodo-payments-product-id-here}"
export REPLICATE_API_TOKEN="${REPLICATE_API_TOKEN:-your-replicate-api-token-here}"
export GOOGLE_GENAI_API_KEY="${GOOGLE_GENAI_API_KEY:-your-google-genai-api-key-here}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-your-gemini-api-key-here}"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Navigate to project root (script should be run from project root)
cd "$(dirname "$0")"

# Start uvicorn server from project root
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

