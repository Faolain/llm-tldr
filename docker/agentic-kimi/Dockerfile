FROM python:3.13-slim-bookworm

ENV PATH="/opt/venv/bin:/root/.local/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        git \
        grep \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /opt/llm-tldr-build

COPY pyproject.toml uv.lock README.md ./
COPY tldr ./tldr
COPY scripts ./scripts

RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python -e ".[dev]" kimi-cli

RUN CLAUDE_BIN="$( \
      /opt/venv/bin/python -c "import claude_agent_sdk; from pathlib import Path; print((Path(claude_agent_sdk.__file__).resolve().parent / '_bundled' / 'claude').as_posix())" \
    )" \
    && ln -sf "${CLAUDE_BIN}" /usr/local/bin/claude

RUN mkdir -p /opt/llm-tldr-src /opt/llm-tldr-src/benchmark /workspace /home/agent \
    && chmod 0777 /workspace /home/agent

COPY docker/agentic-kimi/entrypoint.sh /usr/local/bin/agentic-kimi-entrypoint
RUN chmod +x /usr/local/bin/agentic-kimi-entrypoint

ENTRYPOINT ["/usr/local/bin/agentic-kimi-entrypoint"]
