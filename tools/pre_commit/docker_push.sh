#!/bin/bash
# Build the vllm-openai image and push it to Docker Hub.
# This is a pre-commit hook, but it runs only at the `manual` stage —
# invoke explicitly with: pre-commit run docker-push --hook-stage manual

set -euo pipefail

if ! command -v docker &> /dev/null; then
  echo "Warning: Docker command not found. Skipping image build/push."
  echo "Please install Docker: https://docs.docker.com/get-docker/"
  exit 0
fi
if ! docker info &> /dev/null; then
  echo "Warning: Docker daemon is not running. Skipping image build/push."
  exit 0
fi

if [ -z "${DOCKERHUB_USERNAME:-}" ] || [ -z "${DOCKERHUB_TOKEN:-}" ]; then
  echo "Error: DOCKERHUB_USERNAME and DOCKERHUB_TOKEN must be set." >&2
  echo "  export DOCKERHUB_USERNAME=..." >&2
  echo "  export DOCKERHUB_TOKEN=..." >&2
  exit 1
fi

IMAGE="$DOCKERHUB_USERNAME/vllm-eviction"
TAG="$(git rev-parse --short HEAD)"
STATE_FILE=".docker_build_state"
AGENT_TRACKER_PATHS='^vllm/agent_tracker/|^vllm/entrypoints/openai/agent_tracker/'

echo "Logging in to Docker Hub as $DOCKERHUB_USERNAME..."
docker login --username "$DOCKERHUB_USERNAME" --password-stdin <<< "$DOCKERHUB_TOKEN"

# Use the fast agent_tracker overlay (docker/Dockerfile.agent_tracker) instead of a full
# rebuild when every file that changed since the last full build is under agent_tracker/.
USE_OVERLAY=false
if [ -f "$STATE_FILE" ] && docker image inspect "$IMAGE:latest" &> /dev/null; then
  LAST_FULL_BUILD_SHA="$(cat "$STATE_FILE")"
  if git cat-file -e "$LAST_FULL_BUILD_SHA" 2> /dev/null; then
    CHANGED_FILES="$(git diff --name-only "$LAST_FULL_BUILD_SHA")"
    if [ -z "$CHANGED_FILES" ] || ! printf '%s\n' "$CHANGED_FILES" | command grep -qvE "$AGENT_TRACKER_PATHS"; then
      USE_OVERLAY=true
    fi
  fi
fi

if [ "$USE_OVERLAY" = true ]; then
  echo "Only agent_tracker files changed since last full build; using fast overlay build..."
  docker build -f docker/Dockerfile.agent_tracker \
    --build-arg BASE_IMAGE="$IMAGE:latest" \
    -t "$IMAGE:$TAG" -t "$IMAGE:latest" .
else
  echo "Building $IMAGE:$TAG (target vllm-openai)..."
  docker build -f docker/Dockerfile --target vllm-openai \
    --build-arg CUDA_VERSION=12.8.1 \
    --build-arg max_jobs=40 \
    --build-arg nvcc_threads=1 \
    --build-arg RUN_WHEEL_CHECK=false \
    -t "$IMAGE:$TAG" -t "$IMAGE:latest" .
  echo "$(git rev-parse HEAD)" > "$STATE_FILE"
fi

echo "Pushing $IMAGE:$TAG and $IMAGE:latest..."
docker push "$IMAGE:$TAG"
docker push "$IMAGE:latest"

echo "Done: pushed $IMAGE:$TAG and $IMAGE:latest"
