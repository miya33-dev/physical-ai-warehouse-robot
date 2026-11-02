#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS="$PWD/models"
export OLLAMA_NO_CLOUD=1
if [ -x ./.runtime/ollama/ollama ]; then
  exec ./.runtime/ollama/ollama "$@"
fi
exec ollama "$@"
