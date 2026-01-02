# Use a PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for Geospatial libraries (GDAL, PROJ)
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command
CMD ["bash"]