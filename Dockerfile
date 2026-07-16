# Slim Python base keeps the image small
FROM python:3.12-slim

WORKDIR /app

# A build tool some ML libraries need
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (Docker caches this layer unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Build the vector index at image-build time (downloads the local embedding model;
# needs no API key), so the container is self-contained and ready to serve.
RUN python ingest.py

EXPOSE 8000

# Start the FastAPI server, reachable from outside the container
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
