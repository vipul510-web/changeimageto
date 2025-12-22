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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Download vtracer binary from GitHub releases (faster than compiling)
# Try to download pre-built binary for Linux x86_64
RUN VTRACER_VERSION="0.6.4" && \
    VTRACER_URL="https://github.com/visioncortex/vtracer/releases/download/v${VTRACER_VERSION}/vtracer-${VTRACER_VERSION}-x86_64-unknown-linux-gnu.tar.gz" && \
    echo "Attempting to download vtracer binary from: $VTRACER_URL" && \
    (curl -L -f "$VTRACER_URL" -o /tmp/vtracer.tar.gz && \
     tar -xzf /tmp/vtracer.tar.gz -C /tmp && \
     mv /tmp/vtracer /usr/local/bin/vtracer && \
     chmod +x /usr/local/bin/vtracer && \
     rm /tmp/vtracer.tar.gz && \
     echo "vtracer binary installed successfully") || \
    (echo "Binary download failed, will try Python package installation" && \
     apt-get update && apt-get install -y build-essential pkg-config libssl-dev curl && \
     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal && \
     export PATH="/root/.cargo/bin:${PATH}" && \
     pip install --no-cache-dir vtracer==${VTRACER_VERSION} && \
     echo "vtracer Python package installed successfully")

# Copy requirements and install Python dependencies (excluding vtracer if binary was used)
COPY requirements.txt .
RUN pip install --no-cache-dir $(grep -v "^vtracer" requirements.txt || cat requirements.txt)

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
