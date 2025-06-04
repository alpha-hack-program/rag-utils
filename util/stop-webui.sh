#!/bin/bash

CONTAINER_NAME="open-webui"

# Check if container exists
if podman ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
  echo "🛑 Stopping and removing container '${CONTAINER_NAME}'..."
  podman stop "${CONTAINER_NAME}" > /dev/null
  podman rm "${CONTAINER_NAME}" > /dev/null
  echo "✅ Container '${CONTAINER_NAME}' stopped and removed."
else
  echo "ℹ️ No container named '${CONTAINER_NAME}' is running."
fi
