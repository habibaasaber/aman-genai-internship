# ---- Stage 1: dependency install ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system packages needed for MCP (Node.js + npm for npx).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pre-download the GitHub MCP server so the first startup is fast.
RUN npx -y @modelcontextprotocol/server-github --version || true

# Install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy Node.js + npm by installing it properly in runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Skip copying the pre-seeded npm cache as it fails if empty.

# Copy installed Python packages.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source.
COPY . .

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"]
