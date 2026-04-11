# ═══════════════════════════════════════════════════════════════════════
# HomeOn AI Engine – Python service
# Base: python:3.11-slim (Debian Bookworm)
#
# Key decisions:
#  • face_recognition requires compiling dlib – we install pre-built wheels
#  • EasyOCR and ultralytics are installed from PyPI (pure Python / CUDA optional)
#  • opencv-python-headless avoids GUI dependencies
#  • Model weights are downloaded at first run and cached in a volume
# ═══════════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS base

# System dependencies for dlib, OpenCV, and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python packages (dlib compiles from source – takes ~5 min)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY . .

# Snapshot output directory (mounted as a volume shared with backend)
VOLUME ["/snapshots"]

# Model cache (YOLOv8 weights, EasyOCR models)
VOLUME ["/root/.cache"]

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
