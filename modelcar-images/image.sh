#!/bin/bash

set -euo pipefail

# --- Load .env ---
if [ -f .env ]; then
  source .env
else
  echo "❌ .env file not found"
  exit 1
fi

# --- Load Hugging Face token ---
if [ -f .hf.token ]; then
  HF_TOKEN=$(<.hf.token)
else
  echo "❌ .hf.token file not found"
  exit 1
fi

# --- Defaults ---
BASE_TAG="${DEFAULT_BASE_TAG}"
CONTAINER_FILE="${DEFAULT_CONTAINER_FILE}"
CACHE_FLAG="${DEFAULT_CACHE_FLAG:-}"
REGISTRY="${REGISTRY:-quay.io/atarazana}"
IMAGE_NAME="modelcar-catalog"

# --- Model list from .env ---
MODELS=("${MODELS[@]}")

# --- Extract tag from model name (e.g., multilingual-e5-large) ---
extract_tag() {
  basename "$1"
}

# --- Build ---
build() {
  for MODEL_REPO in "${MODELS[@]}"; do
    TAG=$(extract_tag "$MODEL_REPO")
    LOCAL_IMAGE="${IMAGE_NAME}:${TAG}"

    echo "🔨 Building image for model: $MODEL_REPO"
    podman build ${CACHE_FLAG} \
      -t "${LOCAL_IMAGE}" \
      -f "${CONTAINER_FILE}" . \
      --build-arg MODEL_REPO="${MODEL_REPO}" \
      --build-arg BASE_TAG="${BASE_TAG}" \
      --build-arg HF_TOKEN="${HF_TOKEN}"
    echo "✅ Build complete: ${LOCAL_IMAGE}"
  done
}

# --- Push ---
push() {
  for MODEL_REPO in "${MODELS[@]}"; do
    TAG=$(extract_tag "$MODEL_REPO")
    LOCAL_IMAGE="${IMAGE_NAME}:${TAG}"
    REMOTE_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

    echo "📤 Pushing ${LOCAL_IMAGE} to ${REMOTE_IMAGE}"
    podman tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"
    podman push "${REMOTE_IMAGE}"
    echo "✅ Image pushed: ${REMOTE_IMAGE}"
  done
}

# --- Run ---
run() {
  for MODEL_REPO in "${MODELS[@]}"; do
    TAG=$(extract_tag "$MODEL_REPO")
    LOCAL_IMAGE="${IMAGE_NAME}:${TAG}"
    REMOTE_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

    IMAGE_TO_RUN="${LOCAL_IMAGE}"
    if [[ "${1:-}" == "--remote" ]]; then
      echo "🌐 Pulling remote image ${REMOTE_IMAGE}..."
      podman pull "${REMOTE_IMAGE}"
      IMAGE_TO_RUN="${REMOTE_IMAGE}"
    else
      echo "🚀 Running local image ${LOCAL_IMAGE}..."
    fi

    podman run --rm -it "${IMAGE_TO_RUN}" /bin/bash
  done
}

# --- Help ---
help() {
  echo "Usage: ./image.sh [build|push|run [--remote]|all]"
  echo "  build         Build images for all models"
  echo "  push          Push all images to registry"
  echo "  run           Run all images locally"
  echo "  run --remote  Run remote images (pull from registry)"
  echo "  all           Build, push, and run all locally"
}

# --- Entrypoint ---
case "${1:-}" in
  build) build ;;
  push) push ;;
  run) shift; run "$@" ;;
  all)
    build
    push
    run
    ;;
  *) help ;;
esac
