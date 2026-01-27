FROM python:3.11-slim AS builder

WORKDIR /app

ENV UV_VERSION=0.8.9
# RUN pip install --no-cache-dir uv==${UV_VERSION} -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir uv==${UV_VERSION}

# if you located in China, you can use aliyun mirror to speed up
# RUN sed -i 's@deb.debian.org@mirrors.aliyun.com@g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# if you located in China, you can use aliyun mirror to speed up
# ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV UV_HTTP_TIMEOUT=300
RUN uv sync

FROM python:3.11-slim

WORKDIR /app

# if you located in China, you can use aliyun mirror to speed up
# RUN sed -i 's@deb.debian.org@mirrors.aliyun.com@g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY src ./src
COPY images ./images

# Pre-download default embedding model for pyseekdb
# If you have internet problem, you can set HF_ENDPOINT to https://hf-mirror.com
# ENV HF_ENDPOINT=https://hf-mirror.com
RUN python -c "from pyseekdb import DefaultEmbeddingFunction; \
               emb = DefaultEmbeddingFunction(); \
               emb._download_model_if_not_exists()" && \
    echo "Embedding model downloaded successfully"

RUN mkdir -p /app/data/uploaded

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.runOnSave=false", "src/frontend/chat_ui.py"]
