# syntax=docker/dockerfile:1
# Stage 1: builder — install all Python dependencies
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build tools needed by some packages (e.g. faiss-cpu, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Download CPU torch wheel to disk first, then install everything in ONE pip
# invocation. Single resolver = when tokenizers/transformers declare
# torch>=1.13.0, pip sees the local CPU wheel already queued and does NOT
# fetch the 530 MB CUDA build from PyPI.
RUN sed -i '/^torch\r\?$/d' requirements.txt && \
    pip download --no-deps --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
        --dest /tmp/wheels && \
    pip install --no-cache-dir --prefix=/install \
        /tmp/wheels/torch-*.whl \
        -r requirements.txt

# Stage 2: runtime — lean image, non-root user
FROM python:3.10-slim AS runtime

# libgomp1: OpenMP runtime needed by scikit-learn and faiss-cpu at import time.
# The builder has it via gcc, but the runtime stage does not inherit it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application source and config only — models are mounted as a volume
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml .

# models/ directory placeholder so volume mount works without error
RUN mkdir -p models/stage_a_onnx models/baseline data/faiss_index

# Own the working directory
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# TOGETHER_API_KEY must be passed at runtime via --env or --env-file
# Never bake secrets into the image
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
