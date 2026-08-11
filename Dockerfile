FROM python:3.12-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fetch_secrets.py .
COPY entrypoint.sh .
COPY src/ ./src/

# Set the exec bit here rather than trusting the mode on disk. This package is
# deployed by uploading a zip to the tool-executor, and that round-trip drops
# the bit — the container then dies at start with
# `exec: "./entrypoint.sh": permission denied`. Building it in makes the image
# independent of how the source arrived.
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
