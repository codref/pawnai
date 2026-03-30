# litellm proxy image with pawn-agent CustomLLM adapter
#
# Build from the project root:
#   docker build -f docker/litellm/Dockerfile -t pawn-litellm-proxy .
#
# Run (pawn-agent must be reachable at the api_base configured in litellm_config.yaml):
#   docker run -p 4000:4000 pawn-litellm-proxy
#
# Override config at runtime:
#   docker run -p 4000:4000 -v $(pwd)/litellm_config.yaml:/app/config.yaml pawn-litellm-proxy
#
# Set a proxy master key:
#   docker run -p 4000:4000 -e LITELLM_MASTER_KEY=sk-my-key pawn-litellm-proxy

FROM ghcr.io/berriai/litellm:main-latest

WORKDIR /app

# Minimal package stub so `pawn_agent.core.litellm_custom` is importable.
# Only the three files that the custom handler actually needs are copied —
# no heavy ML dependencies (pyannote, NeMo, etc.) are required here.
COPY pawn_agent/__init__.py            pawn_agent/__init__.py
COPY pawn_agent/core/__init__.py       pawn_agent/core/__init__.py
COPY pawn_agent/core/litellm_custom.py pawn_agent/core/litellm_custom.py

COPY litellm_config.yaml /app/config.yaml

EXPOSE 4000

# The litellm base image sets ENTRYPOINT ["litellm"]
CMD ["--config", "/app/config.yaml", "--port", "4000", "--host", "0.0.0.0"]
