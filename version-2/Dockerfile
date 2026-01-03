FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libgdal-dev g++ git cmake libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL paths for the compiler
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Step 1: Install external dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Copy the project files
COPY . .

# Step 3: Now install your local project (replaces the -e . line)
#RUN pip install --no-cache-dir -e .

CMD ["bash"]