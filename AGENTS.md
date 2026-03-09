# AGENTS.md

## Cursor Cloud specific instructions

### Architecture overview

This is **ChangeImageTo.com** — a free online image editing SaaS. Two services are needed locally:

| Service | Command | Port |
|---------|---------|------|
| FastAPI backend | `uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000` | 8000 |
| Static frontend | `python3 -m http.server 8080 --directory frontend` | 8080 |

The frontend auto-detects local dev when served on `127.0.0.1:8080` and routes API calls to `http://127.0.0.1:8000`.

### Key caveats

- **scikit-image** is imported (`from skimage import ...`) in `backend/main.py` but is **not** listed in `requirements.txt`. It must be installed separately (`pip install scikit-image`).
- System packages needed: `libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 tesseract-ocr tesseract-ocr-eng` (for OpenCV headless and OCR).
- `~/.local/bin` must be on `PATH` for `uvicorn` and other pip-installed scripts.
- The `realesrgan-ncnn-vulkan` binary at the repo root must be executable (`chmod +x`). It ships pre-built for Linux x86_64.
- No `__init__.py` in `backend/` — Python 3 namespace packages handle it.
- External API keys (Gemini, Replicate, Dodo Payments, etc.) are **optional**; the backend degrades gracefully without them.

### Lint / test / build

- No formal linter or test framework is configured in this repo.
- Swagger docs at `http://127.0.0.1:8000/docs` are useful for manual API testing.
- The frontend is plain HTML/CSS/JS with no build step.
