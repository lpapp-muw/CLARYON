FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install
COPY pyproject.toml .
COPY claryon/ claryon/
RUN pip install --no-cache-dir -e ".[all]"

# Copy remaining files
COPY configs/ configs/
COPY tests/ tests/
COPY scripts/ scripts/

ENTRYPOINT ["claryon"]
