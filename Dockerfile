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
    && rm -rf /var/lib/apt/lists/*

# Download Real-ESRGAN NCNN-Vulkan binary (no PyTorch needed!)
# This is much lighter than PyTorch (~50-100MB vs 5-10GB)
RUN mkdir -p /app/realesrgan-models && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip -O /tmp/realesrgan.zip && \
    unzip -q /tmp/realesrgan.zip -d /tmp/realesrgan && \
    find /tmp/realesrgan -name "realesrgan-ncnn-vulkan" -type f -executable -exec mv {} /app/ \; && \
    find /tmp/realesrgan -name "*.param" -o -name "*.bin" | head -20 | xargs -I {} mv {} /app/realesrgan-models/ 2>/dev/null || true && \
    chmod +x /app/realesrgan-ncnn-vulkan && \
    rm -rf /tmp/realesrgan /tmp/realesrgan.zip || echo "Real-ESRGAN binary download failed, will be optional"

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
