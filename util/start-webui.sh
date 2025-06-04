#!/bin/bash

# Set variables
WEBUI_IMAGE="ghcr.io/open-webui/open-webui:main"
CONTAINER_NAME="open-webui"
HOST_PORT=3000
# LLM_HOST=$(ipconfig getifaddr en0)
# LLM_PORT=8888
# LLM_HOST=192.168.50.20
LLM_HOST=host.containers.internal
LLM_PORT=7777


API_KEY="NA"  # Dummy key for OpenAI-compatible APIs
API_BASE_URL="http://${LLM_HOST}:${LLM_PORT}/v1"

echo "➡️ API_BASE_URL=${API_BASE_URL}"

# Run the container
podman run -d \
  --name "${CONTAINER_NAME}" \
  -p ${HOST_PORT}:8080 \
  -e OPENAI_API_BASE_URL="${API_BASE_URL}" \
  -e OPENAI_API_KEY="${API_KEY}" \
  -e OLLAMA_ENABLED=false \
  --restart always \
  -v open-webui:/app/backend/data \
  "${WEBUI_IMAGE}"

echo "✅ Open WebUI started at http://localhost:${HOST_PORT}"
echo "🔗 Connected to local LLM at ${API_BASE_URL}"
