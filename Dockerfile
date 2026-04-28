# CLARYON CPU Docker image.
#
# Build:
#     docker build -t claryon .
#     docker build -t claryon-rad --build-arg INSTALL_RADIOMICS=1 .   # with pyradiomics
#
# Run:
#     docker run --rm claryon list-models
#     docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/Results:/app/Results \
#         claryon run -c configs/example_tabular.yaml

FROM python:3.12-slim

ARG INSTALL_RADIOMICS=0

WORKDIR /app

# System dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata + source first so the heavy pip layer can be cached
# even when configs/tests change.
COPY pyproject.toml README.md ./
COPY claryon/ claryon/

# Install CLARYON with all extras (radiomics intentionally excluded from [all],
# see pyproject.toml comment).
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -e ".[all]"

# Optional: pyradiomics with the build-isolation workaround.
# Triggered by --build-arg INSTALL_RADIOMICS=1.
RUN if [ "$INSTALL_RADIOMICS" = "1" ]; then \
        python -m pip install --no-cache-dir numpy versioneer && \
        python -m pip install --no-cache-dir -e ".[radiomics]" --no-build-isolation; \
    fi

# Copy remaining files (configs, tests, scripts) after the heavy install layer
# so they don't invalidate the pip cache.
COPY configs/ configs/
COPY tests/ tests/
COPY scripts/ scripts/

# Smoke test at build time — fails the build if claryon is broken.
RUN python -c "import claryon; print(f'CLARYON {claryon.__version__} built OK')"

ENTRYPOINT ["claryon"]
CMD ["--help"]
