# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Install system dependencies and Python 3.10
RUN apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    python3.10 -m pip install --upgrade pip

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cuda
ENV NO_TORCH_COMPILE=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Hugging Face login (token will be provided at build time)
ARG HF_TOKEN
RUN if [ -n "$HF_TOKEN" ]; then \
    huggingface-cli login --token $HF_TOKEN --add-to-git-credential; \
    fi

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API
CMD ["python3.10", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"] 