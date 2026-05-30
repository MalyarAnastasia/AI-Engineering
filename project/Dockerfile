FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/artifacts/model.joblib
ENV KMEANS_PATH=/app/artifacts/kmeans.joblib

# Expose port
EXPOSE 8000

# Run the service
CMD ["python", "-m", "src.service.app"]
