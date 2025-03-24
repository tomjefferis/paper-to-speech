# Use the official ROCm PyTorch image
FROM rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.1.1

# Set working directory
WORKDIR /app

# Install system dependencies (including lame for MP3 conversion)
RUN apt-get update && apt-get install -y --no-install-recommends \
    lame \
    ffmpeg \
    libsndfile1 \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn jinja2 python-multipart

# Copy application code
COPY . .

# Create upload directory
RUN mkdir -p uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Expose the FastAPI port
EXPOSE 6123

# Run the FastAPI application
CMD ["python3", "app.py"]
