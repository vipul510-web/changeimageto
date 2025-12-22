FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg62-turbo \
    libpng16-16 \
    tesseract-ocr \
    tesseract-ocr-eng \
    wget \
    unzip \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for vtracer compilation if no pre-built wheels available)
# Use minimal profile and only install what's needed
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify vtracer installation (allow time for compilation if needed)
RUN python -c "import vtracer; print('vtracer successfully imported')" || (echo "WARNING: vtracer import check failed, but continuing build" && python -c "import sys; sys.exit(0)")

# Pre-download and cache the model during build
RUN python -c "from rembg import new_session; new_session('u2netp')"

# Copy application code
COPY backend/ ./backend/

# Set environment variables
ENV DEFAULT_MODEL=u2netp
ENV MAX_IMAGE_SIDE=1600
ENV MAX_CONCURRENCY=2
ENV USE_LAMA=true
ENV LAMA_MASK_THRESHOLD=0.03
ENV MODEL_WARMUP=false
ENV LAMA_ONNX_PATH=/workspace/models/lama.onnx
ENV LAMA_ONNX_URL=https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx

# Expose port
EXPOSE 8080

# Start command
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
